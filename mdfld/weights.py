"""Model files.

- MDFLD v2 landmark model (TorchScript): downloaded once from this repository's releases and checked by SHA-256.
- SAM 3.1 weights: gated on Hugging Face (facebook/sam3.1, SAM License). Request access on that page and log in with
  `hf auth login` (or set HF_TOKEN), or pass a local `sam3.1_multiplex.pt` (also accepted through the MDFLD_SAM3_CKPT variable).
The landmark model is cached in ~/.cache/mdfld (or $MDFLD_HOME); SAM 3.1 in the Hugging Face cache.
"""
import hashlib
import os
from pathlib import Path

# The v2 weights exported to TorchScript (an extra file in release v2.0), so that no network code is needed: outputs match the original
# model exactly on CPU and within 2e-7 on GPU.
MDFLD_V2 = dict(url='https://github.com/zenonLEE/MDFLD/releases/download/v2.0/mdfld_cartoon_v2.torchscript.pt',
                name='mdfld_cartoon_v2.torchscript.pt',
                sha256='e035ff0cbcf884a53792288334523b9fe29e4f58a9eb0f946510391a9bddaf2a')


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


def mdfld_v2(path=None):
    """Path to the MDFLD v2 weights, downloading them on first use."""
    if path:
        return Path(path)
    p = cache_dir() / MDFLD_V2['name']
    if not p.exists():
        from torch.hub import download_url_to_file
        tmp = p.with_name(p.name + '.part')
        download_url_to_file(MDFLD_V2['url'], str(tmp), progress=True)
        if _sha256(tmp) != MDFLD_V2['sha256']:
            tmp.unlink()
            raise RuntimeError(f'checksum mismatch for {MDFLD_V2["url"]}; please retry the download')
        tmp.replace(p)
    return p


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
