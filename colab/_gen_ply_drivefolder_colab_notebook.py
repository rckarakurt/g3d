"""Generate VRCaps_PLY_DriveFolder_Colab.ipynb — GitHub clone + gaze pipeline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORBIT = ROOT / "orbit_coating"

STAGES = [
    ("00_colab_git_setup.md", "## 0. GitHub clone"),
    ("00_colab_git_setup.py", None),
    ("00_ply_styleshot_setup.py", "## 1. Kurulum (StyleShot + xatlas)"),
    ("01b_styleshot_models.py", "## 1b. StyleShot modelleri (HF, ip.bin ~8 GB)"),
    ("02_fetch_ply_drive_folder.py", "## 2. Drive klasorunden PLY indir"),
    ("12_ply_drivefolder_pipeline.py", "## 3. Pipeline (UV + StyleShot + render)"),
    ("13_show_view_strip.py", "## 4. Yan yana onizleme (360° full orbit — arka dahil)"),
]

GAZE_STAGES = [
    ("00_gaze_drive_upload.md", None),
    ("05_upload_unity_dataset_colab.py", "## 5b. Unity dataset — Drive → /content"),
    ("14_gaze_colab_pipeline.py", "## 6. Gaze anchor + acilar + trajectory map"),
    ("00_gaze_composite_colab.md", None),
    ("15_gaze_composite_colab.py", "## 7. Trajectory composite (tum kareler)"),
    ("00_paper_angle_composites_colab.md", None),
    ("16_paper_angle_composites_colab.py", "## 8. Makale — 7 sabit aci sentetik composite"),
]


def read_py(name: str) -> str:
    return (ORBIT / name).read_text(encoding="utf-8")


def cell_md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.splitlines()] + ["\n"],
    }


def cell_code(src: str) -> dict:
    lines = src.splitlines()
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in lines] + (["\n"] if lines else []),
        "outputs": [],
        "execution_count": None,
    }


def main() -> None:
    header = """# VRCaps — Drive PLY + StyleShot + Gaze Trajectory

**Polyp kaynagi:** [polyp PLY klasoru](https://drive.google.com/drive/folders/1P9a6WMqMLzmyvg53fOqZV6FRxA-w3veo)

| Adim | Aciklama |
|------|----------|
| **0** | **GitHub clone** (script gomme yok) |
| 1–1b | StyleShot + HF modelleri |
| 2 | Drive'dan PLY |
| 3–4 | UV + texture + render (**Unity acilari**) + onizleme |
| **5b** | **Unity dataset — Drive → /content** |
| **6** | **Gaze anchor + acilar** |
| **7** | **Composite (view bank + RGB, /content)** |

Kod degisikligi: GitHub'a push → Colab'da **Bolum 0** tekrar calistir (`git pull`).
Varsayilan ciktilar **`/content`** — Drive yalnizca `COPY_TO_DRIVE = True` ile yedek.
"""

    cells = [cell_md(header)]
    for fname, title in STAGES + GAZE_STAGES:
        path = ORBIT / fname
        if fname.endswith(".md"):
            cells.append(cell_md(path.read_text(encoding="utf-8")))
        elif title:
            cells.append(cell_md(title + "\n"))
        if fname.endswith(".py"):
            cells.append(cell_code(read_py(fname)))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }

    for out in (
        ROOT / "VRCaps_PLY_DriveFolder_Colab.ipynb",
        ROOT / "colab_drive_pack" / "VRCaps_PLY_DriveFolder_Colab.ipynb",
    ):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("Wrote:", out)


if __name__ == "__main__":
    main()
