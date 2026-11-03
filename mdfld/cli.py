"""Command line: `mdfld image ...` and `mdfld video ...`."""
import argparse
import json
import os
import sys
import time

import cv2


def _common(p):
    p.add_argument('--out', required=True, help='output JSON (for images: a map from each input path to its result)')
    p.add_argument('--vis', help='overlay image (image) or video (video)')
    p.add_argument('--mode', default='default', choices=['default', 'fast', 'best'],
                   help='eyelid speed / fidelity: best = the v2.1 setup (slowest), default, fast')
    p.add_argument('--no-eyelids', action='store_true', help='landmarks only')
    p.add_argument('--device', help='e.g. cuda, cuda:1 or cpu (default: cuda if available); SAM needs a CUDA device')
    p.add_argument('--landmark-checkpoint', help='landmark model, TorchScript (default: MDFLD v2, downloaded on first use)')
    p.add_argument('--sam-checkpoint', help='sam3.1_multiplex.pt (default: Hugging Face facebook/sam3.1, gated)')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='mdfld', description='MDFLD face landmarks and eyelid curves for images and videos.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    pi = sub.add_parser('image', help='one or more images'); pi.add_argument('inputs', nargs='+', help='image files')
    pi.add_argument('--box', help='face box x0,y0,x1,y1 (single image; default: found with SAM)'); _common(pi)
    pv = sub.add_parser('video', help='a video file'); pv.add_argument('input', help='video file')
    pv.add_argument('--every', type=int, default=1, help='run SAM for the eyelids every N frames and interpolate in between (default 1)')
    pv.add_argument('--no-smooth', action='store_true', help='turn off temporal smoothing')
    pv.add_argument('--min-cutoff', type=float, default=1.0, help='One Euro filter: cut-off at rest, in Hz (lower = smoother)')
    pv.add_argument('--beta', type=float, default=5.0, help='One Euro filter: speed coefficient (higher = keeps fast motion such as blinks)')
    pv.add_argument('--max-frames', type=int); _common(pv)
    a = ap.parse_args(argv)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(out_dir):
        ap.error(f'the folder of --out does not exist: {out_dir}')
    if a.cmd == 'video' and a.every < 1:
        ap.error('--every must be at least 1')

    from .pipeline import MDFLD, draw
    m = MDFLD(mode=a.mode, eyelids=not a.no_eyelids, device=a.device, landmark_checkpoint=a.landmark_checkpoint, sam_checkpoint=a.sam_checkpoint)
    if a.cmd == 'image':
        box = tuple(float(v) for v in a.box.split(',')) if a.box else None
        if box and len(a.inputs) > 1:
            ap.error('--box works with a single image')
        if a.vis and len(a.inputs) > 1:
            os.makedirs(a.vis, exist_ok=True)
        results = {}
        try:
            for path in a.inputs:
                t0 = time.time()
                try:
                    r = results[path] = m.image(path, face_box=box)
                    print(f'{path}: {"no face found" if r["landmarks"] is None else "ok"} ({time.time() - t0:.1f}s)', flush=True)
                except Exception as ex:                                       # keep going; the error is recorded in the output
                    results[path] = {'error': f'{type(ex).__name__}: {ex}'}; print(f'{path}: {results[path]["error"]}', file=sys.stderr)
                    continue
                if a.vis and r['landmarks'] is not None:
                    out = a.vis if len(a.inputs) == 1 else os.path.join(a.vis, os.path.splitext(os.path.basename(path))[0] + '_mdfld.png')
                    try:
                        cv2.imwrite(out, draw(path, r['landmarks'], r['eyelids']))        # drawn on the upright image
                    except Exception as ex:
                        print(f'{path}: could not write {out} ({ex})', file=sys.stderr)
        finally:
            json.dump(results, open(a.out, 'w'))
    else:
        r = m.video(a.input, every=a.every, smooth=not a.no_smooth, min_cutoff=a.min_cutoff, beta=a.beta, vis=a.vis, max_frames=a.max_frames)
        json.dump(r, open(a.out, 'w'))
        print(f'{len(r["frames"])} frames, {sum(f["landmarks"] is not None for f in r["frames"])} with a face')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
