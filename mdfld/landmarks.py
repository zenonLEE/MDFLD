"""68-point landmarks with the MDFLD v2 model (iBUG 300-W layout, 0-based).

The model is a TorchScript file (network and weights together), so no network code is needed to run it.
"""
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from . import weights

SIZE = 256


def box_of(points):
    p = np.asarray(points)
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def _crop_matrix(side, cx, cy):
    s = SIZE / side
    to = (SIZE - 1) / 2.0
    return np.array([[s, 0, to - s * cx], [0, s, to - s * cy], [0, 0, 1]], np.float32)


def load_rgb(image):
    """Path, PIL image or RGB array -> RGB PIL image. Files are turned upright according to their EXIF orientation."""
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, (str, bytes)) or hasattr(image, '__fspath__'):
        return ImageOps.exif_transpose(Image.open(image)).convert('RGB')
    return Image.fromarray(np.asarray(image, np.uint8)).convert('RGB')


def to_bgr(image):
    """Path, PIL image or RGB array -> BGR uint8 array (the network takes BGR input)."""
    return np.ascontiguousarray(np.asarray(load_rgb(image))[..., ::-1])


class Landmarker:
    """MDFLD landmark model.

    lm = Landmarker()
    points = lm(image, face_box)            # (68, 2) array in image pixels
    """

    def __init__(self, checkpoint=None, device=None):
        self.device = str(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        path = weights.mdfld_v2(checkpoint)
        try:
            self.net = torch.jit.load(str(path), map_location=self.device).eval()
        except RuntimeError as ex:
            raise RuntimeError(f'{path} is not a TorchScript model. The package uses mdfld_cartoon_v2.torchscript.pt '
                               '(downloaded automatically); the .pkl checkpoints need the training code.') from ex

    @torch.no_grad()
    def predict_boxes(self, image_bgr, boxes, k=1.1):
        """One pass for each (x0, y0, x1, y1) box: a square crop of side k * max(w, h) -> (n, 68, 2)."""
        crops, mats = [], []
        for x0, y0, x1, y1 in boxes:
            M = _crop_matrix(k * max(x1 - x0, y1 - y0, 1.0), (x0 + x1) / 2, (y0 + y1) / 2)
            crops.append(cv2.warpPerspective(image_bgr, M, dsize=(SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)); mats.append(M)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).to(self.device).float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.net(x).float().cpu().numpy()
        p = ((out + 1) * SIZE - 1) / 2
        res = np.empty_like(p)
        for j, M in enumerate(mats):
            Mi = np.linalg.inv(M)
            res[j] = p[j] @ Mi[:2, :2].T + Mi[:2, 2]
        return res.astype(np.float64)

    def __call__(self, image, box, passes=3, k0=1.0, k=1.1):
        """Landmarks for one face in an image (path, PIL image or RGB array). The first pass uses `box`
        (x0, y0, x1, y1; side scaled by k0); each later pass uses the box of the previous pass's 68 points (side x k).
        k = 1.1 keeps the jaw and eyebrows inside the crop: on the CariFace test set, NME 5.0% against 8.3% with k = 0.9."""
        return self.from_bgr(to_bgr(image), box, passes, k0, k)

    def from_bgr(self, image_bgr, box, passes=3, k0=1.0, k=1.1):
        """Same as calling the object, for a BGR array (e.g. a frame from cv2)."""
        p = self.predict_boxes(image_bgr, [tuple(float(v) for v in box)], k0)[0]
        for _ in range(passes - 1):
            p = self.predict_boxes(image_bgr, [box_of(p)], k)[0]
        return p
