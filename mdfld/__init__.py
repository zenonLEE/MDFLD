"""MDFLD: multi-domain face landmarks (68 points) and lash-free eyelid curves, for images and videos.

    from mdfld import MDFLD
    m = MDFLD()                       # downloads the MDFLD v3 weights on first use (model="v2" for the smaller one); SAM 3.1 needs Hugging Face access
    r = m.image("face.png")           # face box, 68 landmarks, eyelid curves
    v = m.video("clip.mp4", vis="clip_mdfld.mp4")

Paper: Li et al., "Towards Multi-Domain Face Landmark Detection with Synthetic Data from Diffusion Model", ICASSP 2024.
"""
from .landmarks import Landmarker  # noqa: F401
from .pipeline import MDFLD, draw  # noqa: F401

__version__ = '3.0.0'
