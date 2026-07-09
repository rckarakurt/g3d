"""Unity gaze_views export for Colab — geometric orbit + Colab view-bank alignment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _finite(val: object) -> bool:
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError):
        return False


def _gaze_csv_ok(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "view_bank_az_deg" not in fields:
            return False
        for row in reader:
            if _finite(row.get("view_bank_az_deg")):
                return True
    return False


def ensure_geometric_gaze_views(
    dataset_dir: Path,
    *,
    write_plot: bool = False,
    force: bool = False,
    gaze_threshold_deg: float | None = None,
) -> dict:
    """Ensure ``poses/gaze_views.csv`` with Colab-aligned ``view_bank_az_deg``.

    Uses reference-free geometric orbit angles from depth + poses, then maps
    ``view_bank_az_deg = -view_bank_az_raw_deg`` to match Colab polyp turntable
    (``mesh_rotate_y(-az)``, camera fixed at +Z).
    """
    dataset_dir = Path(dataset_dir).resolve()
    csv_path = dataset_dir / "poses" / "gaze_views.csv"

    if not force and _gaze_csv_ok(csv_path):
        meta_path = dataset_dir / "poses" / "focus_anchor.json"
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["skipped"] = True
        meta["gaze_views_csv"] = str(csv_path)
        meta["angle_method"] = meta.get("angle_method", "geometric")
        return meta

    from export_gaze_views import export_gaze_views

    kwargs: dict = {
        "write_plot": write_plot,
        "angle_method": "geometric",
    }
    if gaze_threshold_deg is not None:
        kwargs["gaze_threshold_deg"] = float(gaze_threshold_deg)

    return export_gaze_views(dataset_dir, **kwargs)
