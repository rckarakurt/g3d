"""Download StyleShot / SD1.5 / CLIP weights into /content/StyleShot (Hugging Face)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

STYLESHOT_ROOT = Path("/content/StyleShot")

MODELS: list[tuple[str, Callable[[Path], bool]]] = [
    (
        "Gaojunyao/StyleShot",
        lambda p: (p / "pretrained_weight" / "ip.bin").exists(),
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        lambda p: (p / "model_index.json").exists(),
    ),
    (
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        lambda p: (p / "config.json").exists(),
    ),
]


def _model_ok(path: Path, check: Callable[[Path], bool]) -> bool:
    return path.is_dir() and check(path)


def ip_bin_path(styleshot_root: Path | None = None) -> Path:
    root = Path(styleshot_root or STYLESHOT_ROOT)
    return root / "Gaojunyao/StyleShot/pretrained_weight/ip.bin"


def ensure_styleshot_models(styleshot_root: Path | None = None) -> None:
    """HF -> /content/StyleShot/Gaojunyao/StyleShot/... (cwd must match StyleShot)."""
    from huggingface_hub import snapshot_download

    try:
        from colab_net import configure_colab_ssl

        configure_colab_ssl()
    except ImportError:
        pass

    root = Path(styleshot_root or STYLESHOT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(root)

    try:
        for i, (repo_id, check) in enumerate(MODELS, 1):
            local = Path(repo_id)
            if _model_ok(local, check):
                print(f"[{i}/{len(MODELS)}] hazir: {local}")
                continue
            print(f"[{i}/{len(MODELS)}] HF indiriliyor (~GB): {repo_id}")
            snapshot_download(repo_id, local_dir=str(local))
            if not _model_ok(local, check):
                raise RuntimeError(f"Model indirilemedi: {local}")
            print(f"  OK: {local}")
    finally:
        os.chdir(prev)

    ip = ip_bin_path(root)
    if not ip.exists():
        raise FileNotFoundError(f"ip.bin yok: {ip}")
    print("StyleShot modelleri hazir:", ip)
