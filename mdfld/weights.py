"""Model files.

- MDFLD landmark models (TorchScript): downloaded once from this repository's releases and checked by SHA-256.
  v3 (default) contains DINOv3 ViT-B/16 weights and is distributed under the DINOv3 License (LICENSE_DINOv3.md) in addition
  to the MDFLD terms of use; v2 is the smaller, faster model.
- SAM 3.1 weights: gated on Hugging Face (facebook/sam3.1, SAM License). Request access on that page and log in with
  `hf auth login` (or set HF_TOKEN), or pass a local `sam3.1_multiplex.pt` (also accepted through the MDFLD_SAM3_CKPT variable).
The landmark model is cached in ~/.cache/mdfld (or $MDFLD_HOME); SAM 3.1 in the Hugging Face cache.
"""
import hashlib
import os
import sys
from pathlib import Path

# The weights exported to TorchScript, so that no network code is needed. Both take a [B, 3, 256, 256] BGR crop in [-1, 1] and
# return 68 landmarks in [-1, 1].
# v2: the v2.0 model (an extra file in release v2.0); outputs match the original model exactly on CPU and within 2e-7 on GPU.
MDFLD_V2 = dict(url='https://github.com/zenonLEE/MDFLD/releases/download/v2.0/mdfld_cartoon_v2.torchscript.pt',
                name='mdfld_cartoon_v2.torchscript.pt',
                sha256='e035ff0cbcf884a53792288334523b9fe29e4f58a9eb0f946510391a9bddaf2a')
# v3: the v2 network plus frozen DINOv3 ViT-B/16 features (release v3.0); fp32, outputs match the original model exactly on CPU.
MDFLD_V3 = dict(url='https://github.com/zenonLEE/MDFLD/releases/download/v3.0/mdfld_cartoon_v3.torchscript.pt',
                name='mdfld_cartoon_v3.torchscript.pt',
                sha256='dc5de1817397fb0f8eed113b971ea367e21bf7fb09ab22dff11181bf7cd0a337')
MODELS = {'v3': MDFLD_V3, 'v2': MDFLD_V2}
DEFAULT_MODEL = 'v3'


def cache_dir():
    d = Path(os.environ.get('MDFLD_HOME', Path.home() / '.cache' / 'mdfld'))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def landmark_model(model=DEFAULT_MODEL, path=None):
    """Path to the landmark model file ('v3' or 'v2'): the given path, or a download on first use."""
    if model not in MODELS:
        raise ValueError(f'model must be one of {list(MODELS)}')
    if path:
        return Path(path)
    m = MODELS[model]; p = cache_dir() / m['name']
    if not p.exists():
        if model == 'v3':
            print('mdfld_cartoon_v3 contains DINOv3 weights, distributed under the DINOv3 License: '
                  'https://github.com/zenonLEE/MDFLD/blob/main/LICENSE_DINOv3.md', file=sys.stderr)
        from torch.hub import download_url_to_file
        tmp = p.with_name(p.name + '.part')
        download_url_to_file(m['url'], str(tmp), progress=True)
        if _sha256(tmp) != m['sha256']:
            tmp.unlink()
            raise RuntimeError(f'checksum mismatch for {m["url"]}; please retry the download')
        tmp.replace(p)
    return p


def mdfld_v2(path=None):
    """Path to the MDFLD v2 weights, downloading them on first use (kept for compatibility)."""
    return landmark_model('v2', path)


def sam31(path=None):
    """Path to sam3.1_multiplex.pt: the given path, $MDFLD_SAM3_CKPT, or a Hugging Face download (gated)."""
    path = path or os.environ.get('MDFLD_SAM3_CKPT')
    if path:
        return Path(path)
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id='facebook/sam3.1', filename='sam3.1_multiplex.pt'))
    except Exception as ex:
        raise RuntimeError('Could not get the SAM 3.1 checkpoint. Request access at https://huggingface.co/facebook/sam3.1, '
                           'run `hf auth login` (or set HF_TOKEN), or pass the path of sam3.1_multiplex.pt (sam_checkpoint=... or '
                           '--sam-checkpoint, or the MDFLD_SAM3_CKPT variable).') from ex
