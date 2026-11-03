"""Lash-free, smooth eyelid curves from 68-point landmarks and SAM 3.1 text prompts (the v2.1 method).

For each eye: crop a square around the eye (centre = mean of the 6 eye landmarks, side = max(2 x eye width, 320 px)) a
few times with small shifts / scales, ask SAM for "eye", "white of the eye", "eyelid" and "pupil", average the masks,
remove the lash band (eye minus eyelid, keeping eye white), split the outline into upper / lower lid, snap the lids
to the edge of the eye white, replace a corner by the landmark corner only when SAM stops short of the canthus, and fit
each lid with a robust smoothing spline (60 points). "L" / "R" are the image-left / image-right eye.
"""
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from scipy.interpolate import splprep, splev

IBUG = {'L': dict(upper=[36, 37, 38, 39], lower=[36, 41, 40, 39]), 'R': dict(upper=[42, 43, 44, 45], lower=[42, 47, 46, 45])}
EYE_IDX = {'L': list(range(36, 42)), 'R': list(range(42, 48))}
PROMPTS = (('eye', 'eye'), ('scl', 'white of the eye'), ('lid', 'eyelid'), ('pup', 'pupil'))
KEYS = ('eye', 'scl', 'lid', 'pup')
JIT = ((0, 0, 1.0), (0.03, 0, 1.0), (-0.03, 0, 1.0), (0, 0.03, 0.92), (0, -0.03, 1.08))   # (dx, dy) in eye widths, side scale
BLUNT, GAP, LIDPATH = 0.2, 0.06, 0.5
EW_RANGE = (60.0, 180.0)


# ----------------------------------------------------------------------------- landmark curves
def catmull_rom(P, n=400, alpha=0.5):
    """Centripetal Catmull-Rom through P (one virtual point extrapolated at each end) -> [n, 2]."""
    P = np.asarray(P, np.float64)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t0 = 0.0; t1 = t0 + np.linalg.norm(p1 - p0) ** alpha + 1e-9; t2 = t1 + np.linalg.norm(p2 - p1) ** alpha + 1e-9; t3 = t2 + np.linalg.norm(p3 - p2) ** alpha + 1e-9
        for t in np.linspace(t1, t2, n // (len(Q) - 3), endpoint=(i == len(Q) - 3)):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.asarray(out)


def sample(curve, fr):
    """Points at normalised arc-length positions fr on a polyline."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]); f = s / max(s[-1], 1e-9)
    return np.stack([np.interp(fr, f, curve[:, 0]), np.interp(fr, f, curve[:, 1])], -1)


def eye_polygon(lm, e, shape):
    """Closed eye polygon from the landmark lid curves (60 points each)."""
    cu = sample(catmull_rom(lm[IBUG[e]['upper']]), np.linspace(0, 1, 60)); cl = sample(catmull_rom(lm[IBUG[e]['lower']]), np.linspace(0, 1, 60))
    mk = np.zeros(shape, np.uint8); cv2.fillPoly(mk, [np.round(np.vstack([cu, cl[::-1]])).astype(np.int32)], 1)
    return mk.astype(bool)


def iou(a, b):
    u = np.logical_or(a, b).sum(); return float(np.logical_and(a, b).sum() / u) if u else 0.0


def roi_of(c, ew, shape, k=2.8):
    H, W = shape; s = k * ew
    return (int(max(0, c[0] - s / 2)), int(max(0, c[1] - s / 2)), int(min(W, c[0] + s / 2)), int(min(H, c[1] + s / 2)))


def sub(m, roi):
    return m[roi[1]:roi[3], roi[0]:roi[2]]


def paste_roi(m, box, roi):
    """A crop-sized mask placed at box, cut to the working window roi."""
    out = np.zeros((roi[3] - roi[1], roi[2] - roi[0]), bool)
    x0, y0, x1, y1 = max(box[0], roi[0]), max(box[1], roi[1]), min(box[2], roi[2]), min(box[3], roi[3])
    if x1 > x0 and y1 > y0:
        out[y0 - roi[1]:y1 - roi[1], x0 - roi[0]:x1 - roi[0]] = m[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]]
    return out


def segment_eye(sam, img, lm, e, jitters=JIT):
    """Jittered single-eye crops x four prompts -> per-prompt instance counts in the eye's working window."""
    W, H = img.size; shape = (H, W); pts = lm[EYE_IDX[e]]; c = pts.mean(0); ew = float(np.linalg.norm(pts[3] - pts[0]))
    roi = roi_of(c, max(ew, 320.0 / 2.0), shape); rs = (roi[3] - roi[1], roi[2] - roi[0]); lmr = sub(eye_polygon(lm, e, shape), roi)
    nok = {k: 0 for k in KEYS}
    if min(rs) < 2:                                                           # the eye is outside the image
        return dict(roi=roi, acc=None, nok=nok, ew=ew, pts=pts, lmr=lmr)
    acc = {k: np.zeros(rs, np.float32) for k in KEYS}
    for dx, dy, sk in jitters:
        side = max(2.0 * ew, 320.0) * sk; cc = c + np.array([dx, dy]) * ew
        b = (int(round(cc[0] - side / 2)), int(round(cc[1] - side / 2)), int(round(cc[0] + side / 2)), int(round(cc[1] + side / 2)))
        b = (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
        if b[2] - b[0] < 16 or b[3] - b[1] < 16:                              # crop almost entirely outside the image
            continue
        crop = img.crop(b); sid = sam.open(crop); got = {}
        try:
            for key, text in PROMPTS:
                got[key] = [paste_roi(x, b, roi) for x in sam.ask(sid, text, (crop.size[1], crop.size[0]))]
        finally:
            sam.close(sid)
        ev = [iou(x, lmr) for x in got['eye']]
        if not ev or max(ev) <= 0.05:                                         # this crop found no eye on the landmarks
            continue
        em = got['eye'][int(np.argmax(ev))]; acc['eye'] += em; nok['eye'] += 1
        dil = ndimage.binary_dilation(em, iterations=max(1, int(0.1 * ew)))
        sc_ = [x for x in got['scl'] if x.sum() > 0.04 * em.sum() and (x & dil).sum() / x.sum() > 0.5]
        acc['scl'] += np.any(sc_, 0) if sc_ else 0; nok['scl'] += 1
        ld_ = [x for x in got['lid'] if x.sum() and (x & dil).sum() / x.sum() > 0.1]
        acc['lid'] += np.any(ld_, 0) if ld_ else 0; nok['lid'] += 1
        pu_ = [x for x in got['pup'] if x.sum() and (x & em).sum() / x.sum() > 0.5]
        acc['pup'] += np.any(pu_, 0) if pu_ else 0; nok['pup'] += 1
    return dict(roi=roi, acc=acc, nok=nok, ew=ew, pts=pts, lmr=lmr)


# ----------------------------------------------------------------------------- masks -> contour -> curves
def largest_overlapping(m, ref):
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    ov = ndimage.sum(ref, lab, range(1, n + 1)); sz = ndimage.sum(np.ones_like(ref), lab, range(1, n + 1))
    return lab == int(np.argmax(ov if ov.max() > 0 else sz)) + 1


def masks(seg):
    """Average, blur, threshold; clean the eye white; eyeball = eye - (eyelid - eye white)."""
    ew, lmr, nok = seg['ew'], seg['lmr'], seg['nok']
    soft = {k: ndimage.gaussian_filter(seg['acc'][k] / max(nok[k], 1), 1.0) for k in KEYS}
    eye_m, scl_m, lid_m, pup_m = (soft[k] >= 0.5 for k in KEYS)
    if pup_m.any():
        scl_m = scl_m & ~ndimage.binary_dilation(pup_m, iterations=max(1, int(0.04 * ew)))
    lab, nl = ndimage.label(scl_m)
    if nl:
        sz = ndimage.sum(np.ones_like(scl_m), lab, range(1, nl + 1)); ea = max(int(eye_m.sum()), 1)
        scl_m = np.isin(lab, [i + 1 for i, v in enumerate(sz) if v >= 0.04 * ea])
    lid_use = lid_m & ~scl_m
    if (lid_use & eye_m).sum() > 0.35 * eye_m.sum():                         # eyelid covers the eye: treat as a failure
        lid_use = np.zeros_like(lid_use)
    eb = ndimage.binary_fill_holes(largest_overlapping(eye_m & ~lid_use, lmr))
    r_o = max(1, int(round(0.015 * ew)))
    eb2 = ndimage.binary_opening(eb, structure=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_o + 1, 2 * r_o + 1)).astype(bool))
    eb2 = ndimage.binary_fill_holes(largest_overlapping(eb2, lmr)) if eb2.any() else eb
    return dict(soft=soft, eye=eye_m, scl=scl_m, lid=lid_use, eb=eb2)


def chains(mask, axis, prior, rad):
    """Largest outer contour -> (left corner, right corner, upper chain, lower chain, contour); corners = outermost
    contour points within rad of the landmark corners (nearest point if none)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0].astype(np.float64)
    pr = c @ axis; iL, iR = int(np.argmin(pr)), int(np.argmax(pr)); n = len(c)
    for side, q in ((0, np.asarray(prior[0], float)), (1, np.asarray(prior[1], float))):
        d = np.linalg.norm(c - q, axis=1); near = np.where(d <= rad)[0]
        j = int(near[np.argmin(pr[near])] if side == 0 else near[np.argmax(pr[near])]) if len(near) else int(np.argmin(d))
        iL, iR = (j, iR) if side == 0 else (iL, j)
    a = c[np.r_[iL:iR + 1]] if iL <= iR else c[np.r_[iL:n, 0:iR + 1]]
    b = c[np.r_[iR:iL + 1]][::-1] if iR <= iL else c[np.r_[iR:n, 0:iL + 1]][::-1]
    nrm = np.array([-axis[1], axis[0]])
    up, lo = (a, b) if (a @ nrm).mean() < (b @ nrm).mean() else (b, a)
    return c[iL], c[iR], up, lo, c


def snap_to_white(up, lo, M, Y):
    """Move lid points onto the eye white when a dark gap (upper) or a gap (lower) separates them from it."""
    eb, scl = M['eb'], M['scl']
    win = 0.3 * float(eb.sum(0).max()); sl = float(np.median(Y[scl])) if scl.sum() > 30 else None
    if not sl:
        return up, lo
    up = up.copy()
    for i in range(len(up)):
        x, y = int(round(up[i, 0])), int(round(up[i, 1]))
        if not (0 <= x < scl.shape[1]):
            continue
        col = np.where(scl[y:, x])[0]
        if len(col) and 1.5 < col[0] <= win:
            ys = y + int(col[0]); gap = Y[y:ys, x]
            if len(gap) and float(gap.mean()) < 0.6 * sl:
                up[i, 1] = ys
    lo = lo.copy()
    for i in range(len(lo)):
        x, y = int(round(lo[i, 0])), int(round(lo[i, 1]))
        if not (0 <= x < scl.shape[1]) or y <= 0:
            continue
        col = np.where(scl[:y + 1, x])[0]
        if len(col) and 1.5 < (y - col[-1]) <= win:
            lo[i, 1] = col[-1]
    return up, lo


def fit_curve(pts, pL, pR, sigma=1.0, iters=2):
    """Smoothing spline pinned at both corners, two rounds of robust reweighting -> 60 points."""
    P = np.vstack([pL, pts[1:-1], pR]) if len(pts) > 2 else np.vstack([pL, pR])
    P = P[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]]
    if len(P) < 5:
        return sample(np.vstack([pL, pR]), np.linspace(0, 1, 60))
    w = np.ones(len(P)); w[0] = w[-1] = 30.0
    try:
        for _ in range(iters + 1):
            tck, u = splprep([P[:, 0], P[:, 1]], w=w, s=len(P) * sigma ** 2, k=3)
            r = np.linalg.norm(np.stack(splev(u, tck), 1) - P, axis=1)
            w = np.where(r > 2.5 * sigma, 2.5 * sigma / np.maximum(r, 1e-6), 1.0); w[0] = w[-1] = 30.0
        dense = np.stack(splev(np.linspace(0, 1, 600), tck), 1)
    except Exception:
        dense = P
    return sample(dense, np.linspace(0, 1, 60))


def fit_dense(P, sigma=1.0, iters=2, per=False, n=1200):
    """Same robust spline, open (ends pinned) or closed (periodic); dense points."""
    P = P[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]]
    if per:
        while len(P) > 1 and np.linalg.norm(P[-1] - P[0]) < 1e-6:
            P = P[:-1]
        P = np.vstack([P, P[:1]])
    w = np.ones(len(P))
    if not per:
        w[0] = w[-1] = 30.0
    for _ in range(iters + 1):
        tck, u = splprep([P[:, 0], P[:, 1]], w=w, s=len(P) * sigma ** 2, k=3, per=int(per))
        r = np.linalg.norm(np.stack(splev(u, tck), 1) - P, axis=1)
        w = np.where(r > 2.5 * sigma, 2.5 * sigma / np.maximum(r, 1e-6), 1.0)
        if not per:
            w[0] = w[-1] = 30.0
    return np.stack(splev(np.linspace(0, 1, n), tck), 1)


def end_width(c, ax, ew, side, delta=0.05):
    """Spread across the corner axis of the contour points within delta * ew of one end, / ew."""
    sm = np.stack([ndimage.gaussian_filter1d(c[:, i], 2, mode='wrap') for i in (0, 1)], 1)
    pr = sm @ ax; nrm = np.array([-ax[1], ax[0]])
    sel = pr <= pr.min() + delta * ew if side == 0 else pr >= pr.max() - delta * ew
    q = sm[sel] @ nrm
    return float(q.max() - q.min()) / ew


def fit_lids(up, lo, pl, pr_, blunt, ax):
    """Sharp ends pinned; a blunt end is fitted through and split at its outermost point."""
    if not any(blunt):
        return fit_curve(up, pl, pr_), fit_curve(lo, pl, pr_)
    U, Lw = np.vstack([pl, up[1:-1], pr_]), np.vstack([pl, lo[1:-1], pr_]); nrm = np.array([-ax[1], ax[0]])
    try:
        if all(blunt):
            d = fit_dense(np.vstack([U, Lw[::-1][1:-1]]), per=True); iL, iR = int(np.argmin(d @ ax)), int(np.argmax(d @ ax))
            a1 = d[iL:iR + 1] if iL <= iR else np.vstack([d[iL:], d[:iR + 1]])
            a2 = (d[iR:iL + 1] if iR <= iL else np.vstack([d[iR:], d[:iL + 1]]))[::-1]
            cu, cl = (a1, a2) if (a1 @ nrm).mean() < (a2 @ nrm).mean() else (a2, a1)
        elif blunt[1]:
            d = fit_dense(np.vstack([U, Lw[::-1][1:]])); k = int(np.argmax(d @ ax)); cu, cl = d[:k + 1], d[k:][::-1]
        else:
            d = fit_dense(np.vstack([U[::-1], Lw[1:]])); k = int(np.argmin(d @ ax)); cu, cl = d[:k + 1][::-1], d[k:]
    except Exception:
        return fit_curve(up, pl, pr_), fit_curve(lo, pl, pr_)
    return sample(cu, np.linspace(0, 1, 60)), sample(cl, np.linspace(0, 1, 60))


def eye_curves(seg, Y):
    """Masks -> upper / lower lid curves (60 points each, image coordinates) + what was decided."""
    roi, ew, pts = seg['roi'], seg['ew'], seg['pts']
    if seg['nok']['eye'] == 0:
        return None
    M = masks(seg); Yr = sub(Y, roi); off = np.array([roi[0], roi[1]], np.float64)
    if not M['eb'].any():
        return None
    axis = (pts[3] - pts[0]) / max(ew, 1e-6); ax = axis if axis[0] > 0 else -axis
    pl, pr_, up, lo, c = chains(M['eb'], ax, prior=(pts[0] - off, pts[3] - off), rad=0.15 * ew)
    up, lo = snap_to_white(up, lo, M, Yr)
    ring = ndimage.binary_dilation(M['eye'], iterations=max(1, int(0.4 * ew))) & ~ndimage.binary_dilation(M['eye'], iterations=max(1, int(0.1 * ew)))
    skin = float(np.median(Yr[ring])) if ring.sum() > 50 else float(np.median(Yr))
    H, W = Yr.shape; lidd = ndimage.binary_dilation(M['lid'], iterations=2) if M['lid'].any() else M['lid']; used = []
    for side, q in (('L', pts[0] - off), ('R', pts[3] - off)):
        p = pl if side == 'L' else pr_
        if not (float(((p - q) if side == 'L' else (q - p)) @ ax) > 0 and np.linalg.norm(p - q) <= 0.25 * ew):
            continue                                                          # landmark corner not further out, or too far
        x, y = int(round(q[0])), int(round(q[1])); inside = 0 <= x < W and 0 <= y < H
        patch = Yr[max(0, y - 2):y + 3, max(0, x - 2):x + 3]; dark = patch.size > 0 and float(patch.min()) < 0.75 * skin
        t = np.linspace(0, 1, max(3, int(np.linalg.norm(q - p) * 2)))[1:]; seg_ = p[None] + t[:, None] * (q - p)[None]
        sx, sy = np.clip(np.round(seg_[:, 0]).astype(int), 0, W - 1), np.clip(np.round(seg_[:, 1]).astype(int), 0, H - 1)
        gap, in_eye, seg_lid = float(np.linalg.norm(q - p)) / ew, bool(inside and M['eye'][y, x]), float(lidd[sy, sx].mean())
        if dark and not in_eye and round(gap, 3) >= GAP and round(seg_lid, 2) >= LIDPATH:
            up = up[np.linalg.norm(up - p, axis=1) > 0.1 * ew]; lo = lo[np.linalg.norm(lo - p, axis=1) > 0.1 * ew]; used.append(side)
            if side == 'L':
                pl = q
            else:
                pr_ = q
    up = np.vstack([pl, up, pr_]) if len(up) else np.vstack([pl, pr_]); lo = np.vstack([pl, lo, pr_]) if len(lo) else np.vstack([pl, pr_])
    blunt = tuple(s not in used and round(end_width(c, ax, ew, i), 3) > BLUNT for i, s in enumerate('LR'))
    cu, cl = fit_lids(up, lo, pl, pr_, blunt, ax)
    return dict(upper=cu + off, lower=cl + off, landmark_corners=used, blunt_ends=[s for s, b in zip('LR', blunt) if b])


def eyelids(sam, image, lm, rescale=True, jitters=JIT):
    """One face: image (PIL RGB) + 68 landmarks -> {'L': {...} or None, 'R': ...} in image coordinates."""
    lm = to68(lm); ox = oy = 0; sx = sy = 1.0
    ew = np.mean([np.linalg.norm(lm[EYE_IDX[e]][3] - lm[EYE_IDX[e]][0]) for e in 'LR'])
    if rescale and not (EW_RANGE[0] <= ew <= EW_RANGE[1]):
        s = (100.0 if ew < EW_RANGE[0] else 120.0) / ew
        m = 0.6 * max(2.0 * ew, 320.0 / s) * 1.08 + 0.1 * ew                 # room for the largest eye crop, in input pixels
        x0, y0 = np.floor(lm.min(0) - m).astype(int); x1, y1 = np.ceil(lm.max(0) + m).astype(int)
        ox, oy = max(0, x0), max(0, y0); image = image.crop((ox, oy, min(image.size[0], x1), min(image.size[1], y1)))
        W, H = max(1, int(round(image.size[0] * s))), max(1, int(round(image.size[1] * s))); sx, sy = W / image.size[0], H / image.size[1]
        image = image.resize((W, H), Image.LANCZOS)
        lm = np.stack([(lm[:, 0] - ox + 0.5) * sx - 0.5, (lm[:, 1] - oy + 0.5) * sy - 0.5], 1)    # pixel centres
    back = (lambda q: q) if sx == sy == 1.0 and ox == oy == 0 else (lambda q: np.stack([(q[:, 0] + 0.5) / sx - 0.5 + ox, (q[:, 1] + 0.5) / sy - 0.5 + oy], 1))
    Y = np.asarray(image.convert('L'), np.float32); out = {}
    for e in 'LR':
        r = eye_curves(segment_eye(sam, image, lm, e, jitters), Y)
        out[e] = None if r is None else dict(upper=np.round(back(r['upper']), 2).tolist(), lower=np.round(back(r['lower']), 2).tolist(),
                                              landmark_corners=r['landmark_corners'], blunt_ends=r['blunt_ends'])
    return out


def to68(a):
    a = np.asarray(a, np.float64).reshape(-1, 2)
    if len(a) != 68:
        raise ValueError(f'expected 68 landmarks, got {len(a)}')
    return a
