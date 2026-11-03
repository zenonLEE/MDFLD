"""SAM 3.1 text-prompt segmentation, used to find faces and to segment eyes (needs a CUDA GPU).

Two backends share the SAM 3.1 weights:
- "image": SAM 3's image API (Sam3Processor). Each crop is encoded once and all prompts reuse the encoding. Fast.
- "video": the official SAM 3.1 video predictor with each crop as a one-frame video. Slower; this is the setup the
  eyelid method was developed with, so mode="best" reproduces the v2.1 results exactly.
Both run SAM in bfloat16 with TF32, as the official predictor does, without changing these settings for the caller.
"""
import contextlib
import io
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from . import weights


def _require_sam3(device):
    try:
        import sam3  # noqa: F401
        import sam3.model_builder  # noqa: F401
    except ImportError as ex:
        raise ImportError(f'SAM 3 could not be imported ({ex}). Install the extra: pip install "mdfld[sam] @ git+https://github.com/zenonLEE/MDFLD". '
                          'Landmarks alone need no SAM: pass a face box and eyelids=False (--box ... --no-eyelids).') from ex
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('SAM 3.1 needs a CUDA GPU. Without one, pass a face box and eyelids=False (--box ... --no-eyelids).')


@contextlib.contextmanager
def _sam_precision(device):
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        with torch.cuda.device(device), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


@contextlib.contextmanager
def _restore_autocast_state():
    """The official predictor enters bfloat16 autocast for the whole process when it is built (and never leaves it)
    and turns on TF32. Undo all of that, so that the caller's own code is unaffected."""
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    enabled, dtype, cache = torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'), torch.is_autocast_cache_enabled()
    depth = torch.autocast_increment_nesting() - 1; torch.autocast_decrement_nesting()
    try:
        yield
    finally:
        now = torch.autocast_increment_nesting() - 1; torch.autocast_decrement_nesting()
        for _ in range(max(0, now - depth)):
            torch.autocast_decrement_nesting()
        torch.set_autocast_enabled('cuda', enabled); torch.set_autocast_dtype('cuda', dtype); torch.set_autocast_cache_enabled(cache)
        if depth == 0:
            torch.clear_autocast_cache()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


class ImageSam:
    """SAM 3 image API with the SAM 3.1 weights. open(img) -> state; ask(state, text) -> list of bool masks."""

    def __init__(self, checkpoint=None, threshold=0.5, device='cuda'):
        self.device = str(device); _require_sam3(self.device)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        with contextlib.redirect_stdout(io.StringIO()), torch.cuda.device(self.device):   # it reports the unused neck level of 3.1
            model = build_sam3_image_model(checkpoint_path=str(weights.sam31(checkpoint)), load_from_HF=False, device=self.device)
        self.proc = Sam3Processor(model, device=self.device, confidence_threshold=threshold)

    def open(self, img):
        with _sam_precision(self.device):
            return self.proc.set_image(img)

    def ask(self, state, text, shape=None):
        self.proc.reset_all_prompts(state)
        with _sam_precision(self.device):
            out = self.proc.set_text_prompt(text, state)
        m = out.get('masks')
        return [x for x in m[:, 0].float().cpu().numpy().astype(bool)] if m is not None and len(m) else []

    def ask_scored(self, state, text):
        """(score, box (x0, y0, x1, y1)) for each instance, best first."""
        self.proc.reset_all_prompts(state)
        with _sam_precision(self.device):
            out = self.proc.set_text_prompt(text, state)
        if out.get('masks') is None or not len(out['masks']):
            return []
        sc = out['scores'].float().cpu().numpy(); bx = out['boxes'].float().cpu().numpy()
        return sorted(zip(sc.tolist(), [tuple(float(v) for v in b) for b in bx]), key=lambda t: -t[0])

    def close(self, state):
        pass


class VideoSam:
    """Official SAM 3.1 video predictor, one crop = one frame (the setup of the v2.1 eyelid results)."""

    def __init__(self, checkpoint=None, device='cuda'):
        self.device = str(device); _require_sam3(self.device)
        from sam3.model_builder import build_sam3_predictor
        with _restore_autocast_state(), torch.cuda.device(self.device):
            self.pred = build_sam3_predictor(version='sam3.1', checkpoint_path=str(weights.sam31(checkpoint)), use_fa3=False, async_loading_frames=False)
        self.tmp = Path(tempfile.mkdtemp(prefix='mdfld_sam_')); self.dirs = {}

    def open(self, img):
        # sam3 @ f1647e8: start_session passes offload_state_to_cpu, which the 3.1 multiplex init_state rejects,
        # so the session is opened by calling init_state directly.
        fd = self.tmp / uuid.uuid4().hex; fd.mkdir(); path = fd / '000000.jpg'; img.save(path, quality=95)
        try:
            with _sam_precision(self.device):
                st = self.pred.model.init_state(resource_path=str(path), offload_video_to_cpu=False, async_loading_frames=False)
        except Exception:
            shutil.rmtree(fd, ignore_errors=True)
            raise
        sid = str(uuid.uuid4()); now = time.time()
        self.pred._all_inference_states[sid] = {'state': st, 'session_id': sid, 'start_time': now, 'last_use_time': now}
        self.dirs[sid] = fd
        return sid

    def ask(self, sid, text, shape):
        with _sam_precision(self.device):
            self.pred.handle_request({'type': 'reset_session', 'session_id': sid})
            o = self.pred.handle_request({'type': 'add_prompt', 'session_id': sid, 'frame_index': 0, 'text': text})['outputs']
        m = o.get('out_binary_masks')
        if m is None:
            return []
        m = (m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)).astype(bool)
        if m.ndim == 4:
            m = m[:, 0]
        if len(m) and m.shape[1:] != shape:
            m = np.stack([cv2.resize(x.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for x in m])
        return list(m)

    def close(self, sid):
        try:
            with _sam_precision(self.device):
                self.pred.handle_request({'type': 'close_session', 'session_id': sid})
        finally:
            shutil.rmtree(self.dirs.pop(sid, ''), ignore_errors=True)

    def __del__(self):
        shutil.rmtree(getattr(self, 'tmp', ''), ignore_errors=True)


def find_faces(sam, image):
    """All "face" instances in a PIL image as (score, box), best first."""
    st = sam.open(image)
    try:
        return sam.ask_scored(st, 'face')
    finally:
        sam.close(st)
