"""Image and video pipeline: face -> 68 landmarks -> eyelid curves. One face per image or frame."""
import math
import time
import warnings

import cv2
import numpy as np
from PIL import Image

from . import eyelids as E
from .landmarks import Landmarker, load_rgb, to_bgr
from .smooth import smooth_sequence

MODES = {                         # SAM backend, single-eye crops per eye; times per 1404 x 1404 image incl. face + landmarks
    'best': ('video', E.JIT),     # the v2.1 setup: identical eyelids, about 5.0 s
    'default': ('image', E.JIT),  # SAM image API: about 2.5 s, eyelids within about 1 px of "best" (median)
    'fast': ('image', E.JIT[:3]),  # 3 crops per eye: about 1.9 s
}
CORNERS = {'L': (36, 39), 'R': (42, 45)}
EYE = {'L': list(range(36, 42)), 'R': list(range(42, 48))}
# Sanity bounds for a SAM eyelid curve in the eye's corner frame (unit = corner distance): width, centre along the corner
# axis, offset of the mean height from the eye landmarks, and height relative to the landmark eye height. The curves of
# the 34 eyes the method was developed on fall well inside (0.87-1.00, 0.46-0.53, within 0.03, up to 1.26); outside,
# SAM has usually outlined only the iris, or the brow or the whole eye socket.
PLAUSIBLE = dict(width=(0.75, 1.35), centre=(0.35, 0.65), offset=0.15, height=1.8)


def _round(a):
    return None if a is None else np.round(np.asarray(a, np.float64), 2).tolist()


def landmark_curves(lm, e):
    """Lid curves interpolated through the 6 eye landmarks (60 points each), used when SAM finds no eye."""
    return {k: E.sample(E.catmull_rom(np.asarray(lm)[E.IBUG[e][k]]), np.linspace(0, 1, 60)) for k in ('upper', 'lower')}


def plausible(lm, e, lid):
    """Whether a SAM eyelid curve agrees with the eye landmarks (see PLAUSIBLE)."""
    o, R, s = _eye_frame(lm, e)
    q = (np.vstack([lid['upper'], lid['lower']]) - o) @ R / s
    eye = (np.asarray(lm, np.float64)[EYE[e]] - o) @ R / s
    lo, hi = q[:, 0].min(), q[:, 0].max()
    return (PLAUSIBLE['width'][0] <= hi - lo <= PLAUSIBLE['width'][1]
            and PLAUSIBLE['centre'][0] <= (hi + lo) / 2 <= PLAUSIBLE['centre'][1]
            and abs(q[:, 1].mean() - eye[:, 1].mean()) <= PLAUSIBLE['offset']
            and np.ptp(q[:, 1]) <= PLAUSIBLE['height'] * max(np.ptp(eye[:, 1]), 0.05))


def _checked(lm, got):
    """SAM curves that pass the sanity check (source "sam"); other eyes get the landmark curves (source "landmarks")."""
    out = {}
    for e in 'LR':
        v = got.get(e)
        if v and plausible(lm, e, v):
            out[e] = dict(v, source='sam')
        else:
            out[e] = dict(landmark_curves(lm, e), source='landmarks', landmark_corners=['L', 'R'], blunt_ends=[])
    return out


def _iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def _same_face(a, b, size_tol=0.35, shift_tol=0.5):
    sa, sb = max(a[2] - a[0], a[3] - a[1]), max(b[2] - b[0], b[3] - b[1])
    ca, cb = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]), np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return abs(sb / max(sa, 1e-6) - 1) <= size_tol and np.linalg.norm(cb - ca) <= shift_tol * sa


def _eye_frame(lm, e):
    """Similarity frame of an eye from its two corners: origin, rotation (2 x 2), scale."""
    p0, p1 = np.asarray(lm[CORNERS[e][0]], np.float64), np.asarray(lm[CORNERS[e][1]], np.float64)
    d = p1 - p0; s = max(float(np.linalg.norm(d)), 1e-6); c, n = d / s, np.array([-d[1], d[0]]) / s
    return p0, np.stack([c, n], 1), s


class MDFLD:
    """Face landmarks and eyelid curves for images and videos (one face per image or frame).

    m = MDFLD()                      # mode: "default", "fast" or "best"; eyelids=False for landmarks only
    r = m.image("face.png")          # {"face_box", "landmarks" (68 x 2), "eyelids": {"L": {...}, "R": {...}}}
    v = m.video("clip.mp4")          # {"fps", "frames": [...]}, smoothed over time
    """

    def __init__(self, mode='default', eyelids=True, device=None, landmark_checkpoint=None, sam_checkpoint=None, model=None):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {list(MODES)}')
        self.mode, self.with_eyelids, self.sam_checkpoint = mode, eyelids, sam_checkpoint
        self.landmarker = Landmarker(landmark_checkpoint, device, model)       # model: "v3" (default) or "v2"
        self.device = self.landmarker.device
        self.backend, self.jitters = MODES[mode]
        self._finder = self._eyes = None

    # SAM is loaded on first use, so landmarks with a given face box need no SAM at all.
    @property
    def finder(self):
        if self._finder is None:
            from .sam import ImageSam
            self._finder = ImageSam(self.sam_checkpoint, device=self.device)
        return self._finder

    @property
    def eye_sam(self):
        if self._eyes is None:
            if self.backend == 'image':
                self._eyes = self.finder
            else:
                from .sam import VideoSam
                self._eyes = VideoSam(self.sam_checkpoint, device=self.device)
        return self._eyes

    def find_faces(self, image):
        """All faces SAM finds, as (score, (x0, y0, x1, y1)), best first."""
        from .sam import find_faces
        return find_faces(self.finder, load_rgb(image))

    def eyelids(self, image, landmarks):
        """Eyelid curves for one face with known landmarks. An eye that SAM misses, or whose curve fails the sanity
        check, gets the curves through its 6 landmarks; each eye's "source" says which ("sam" or "landmarks")."""
        lm = E.to68(landmarks)
        return _checked(lm, E.eyelids(self.eye_sam, load_rgb(image), lm, jitters=self.jitters))

    def image(self, image, face_box=None, landmarks=None):
        """One image (path, PIL image or RGB array; files are turned upright by their EXIF orientation).
        Give face_box (x0, y0, x1, y1) or landmarks to skip those steps."""
        img = load_rgb(image)
        if landmarks is None:
            if face_box is None:
                faces = self.find_faces(img)
                face_box = faces[0][1] if faces else None
            if face_box is None:
                return {'face_box': None, 'landmarks': None, 'eyelids': None}
            landmarks = self.landmarker(img, face_box)
        lm = E.to68(landmarks)
        lids = self.eyelids(img, lm) if self.with_eyelids else None
        return {'face_box': _round(face_box), 'landmarks': _round(lm), 'eyelids': _pack(lids)}

    def _lids(self, rgb, lm):
        """Eyelids for one video frame, as eyelids(); an error also falls back to the landmark curves."""
        try:
            got = E.eyelids(self.eye_sam, rgb, lm, jitters=self.jitters)
        except Exception as ex:                                               # keep the video going
            warnings.warn(f'eyelids failed on a frame ({type(ex).__name__}: {ex}); using the landmark curves')
            got = {'L': None, 'R': None}
        return _checked(lm, got)

    def video(self, path, every=1, smooth=True, min_cutoff=1.0, beta=5.0, d_cutoff=2.0, vis=None, max_frames=None, verbose=True):
        """A video file. Every frame is processed like an image: the face is found with SAM (the one that overlaps the
        previous frame's face most), then the landmarks. Eyelids are computed with SAM every `every` frames and on the
        first and last frame of each shot, and interpolated in between in the eye's own frame (so they follow the head).
        Then each shot is smoothed with a One Euro filter run forwards and backwards and averaged, so there is no lag:
        min_cutoff (Hz) sets the smoothing at rest, beta how quickly it lets fast motion such as blinks through.
        `vis` writes an overlay video.
        """
        if every < 1:
            raise ValueError('every must be at least 1')
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f'cannot open {path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and math.isfinite(fps) and 1 <= fps <= 240 else 30.0
        raw, seg, prev_box, last, t0, i, W, H = [], -1, None, None, time.time(), 0, 0, 0

        def close_shot():
            if self.with_eyelids and last is not None and raw[last[0]]['lids'] is None:
                raw[last[0]]['lids'] = self._lids(*last[1:])
        try:
            while max_frames is None or i < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                H, W = bgr.shape[:2]; rgb = Image.fromarray(bgr[..., ::-1])
                faces = self.find_faces(rgb); box = None
                if faces:
                    box = faces[0][1]
                    if prev_box is not None:
                        near = max(faces, key=lambda f: _iou(f[1], prev_box))
                        box = near[1] if _iou(near[1], prev_box) > 0.3 else box
                if box is None:
                    close_shot(); raw.append(None); prev_box = last = None
                else:
                    start = prev_box is None or not _same_face(prev_box, box)        # a new shot: another face or a cut
                    if start:
                        close_shot()
                    seg += start; prev_box = box
                    lm = self.landmarker.from_bgr(bgr, box)
                    key = self.with_eyelids and (start or i % every == 0)
                    raw.append(dict(seg=seg, box=np.asarray(box, np.float64), lm=lm, lids=self._lids(rgb, lm) if key else None))
                    last = (len(raw) - 1, rgb, lm)
                if verbose and i % 25 == 0:
                    print(f'frame {i}: {(time.time() - t0) / (i + 1):.2f} s/frame', flush=True)
                i += 1
            close_shot()
        finally:
            cap.release()
        frames = [{'frame': k, 'face_box': None, 'landmarks': None, 'eyelids': None} for k in range(len(raw))]
        for s in range(seg + 1):
            idx = [k for k, r in enumerate(raw) if r is not None and r['seg'] == s]
            LMr = np.stack([raw[k]['lm'] for k in idx]); BX = np.stack([raw[k]['box'] for k in idx])
            ew = max(float(np.mean([np.linalg.norm(LMr[0][c1] - LMr[0][c0]) for c0, c1 in CORNERS.values()])), 1.0)
            sm = (lambda X: smooth_sequence(X, fps, min_cutoff, beta, d_cutoff, scale=ew)) if smooth else (lambda X: X)
            lids = {e: _fill(e, [raw[k] for k in idx]) for e in 'LR'} if self.with_eyelids else {}
            LM, BX = sm(LMr), sm(BX)
            for e in lids:
                lids[e] = (sm(lids[e][0]), lids[e][1])
            for n, k in enumerate(idx):
                fr = frames[k]; fr['face_box'], fr['landmarks'] = _round(BX[n]), _round(LM[n])
                if self.with_eyelids:
                    fr['eyelids'] = {}
                    for e in 'LR':
                        C, src = lids[e]; ref = raw[k]['lids'][e] if raw[k]['lids'] else {}
                        fr['eyelids'][e] = dict(upper=_round(C[n][:60]), lower=_round(C[n][60:]), source=src[n],
                                                landmark_corners=ref.get('landmark_corners', []), blunt_ends=ref.get('blunt_ends', []))
        if vis:
            _write_vis(path, vis, frames, fps, W, H)
        return {'fps': fps, 'width': W, 'height': H, 'mode': self.mode, 'every': every, 'smooth': smooth, 'frames': frames}


def _fill(e, shot):
    """Curves (upper + lower, 120 points) of one eye for every frame of a shot: frames with SAM results as computed,
    frames in between interpolated linearly in time in the eye's corner frame and mapped back with each frame's corners."""
    T = len(shot); C = np.zeros((T, 120, 2)); src = [''] * T; keys = [n for n, r in enumerate(shot) if r['lids'] is not None]
    local = {}
    for n in keys:
        v = shot[n]['lids'][e]; C[n] = np.vstack([v['upper'], v['lower']]); src[n] = v['source']
        o, R, s = _eye_frame(shot[n]['lm'], e); local[n] = (C[n] - o) @ R / s
    for a, b in zip(keys, keys[1:]):
        for n in range(a + 1, b):
            o, R, s = _eye_frame(shot[n]['lm'], e); t = (n - a) / (b - a)
            C[n] = o + s * ((local[a] + (local[b] - local[a]) * t) @ R.T); src[n] = 'interpolated'
    return C, src


def _write_vis(path, out, frames, fps, W, H):
    cap = cv2.VideoCapture(str(path)); writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise IOError(f'cannot write {out}')
    try:
        for fr in frames:
            ok, bgr = cap.read()
            if not ok:
                break
            writer.write(draw(bgr, fr['landmarks'], fr['eyelids']) if fr['landmarks'] is not None else bgr)
    finally:
        cap.release(); writer.release()


def _pack(lids):
    if lids is None:
        return None
    return {e: None if v is None else dict(v, upper=_round(v['upper']), lower=_round(v['lower'])) for e, v in lids.items()}


def draw(image, landmarks=None, lids=None):
    """Overlay on a BGR array (or a path / PIL image, returned as BGR): landmarks as dots, eyelid curves as lines."""
    vis = image.copy() if isinstance(image, np.ndarray) else to_bgr(image)
    if landmarks is not None:
        for x, y in np.asarray(landmarks):
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 200, 0), -1, cv2.LINE_AA)
    for e, v in (lids or {}).items():
        if v:
            for k in ('upper', 'lower'):
                cv2.polylines(vis, [np.round(np.asarray(v[k])).astype(np.int32)], False, (255, 230, 0), 2, cv2.LINE_AA)
    return vis
