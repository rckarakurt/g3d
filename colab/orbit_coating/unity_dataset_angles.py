"""Unity gaze_views export for Colab — geometric orbit + Colab view-bank alignment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ANGLE_METHOD = "geometric"
ALIGN_TOL_DEG = 2.5


def _finite(val: object) -> bool:
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError):
        return False


def _read_gaze_rows(csv_path: Path) -> tuple[list[str], list[dict]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def is_colab_aligned_gaze_export(dataset_dir: Path) -> bool:
    """True when gaze_views.csv has geometric + Colab-negated bank azimuths."""
    dataset_dir = Path(dataset_dir).resolve()
    csv_path = dataset_dir / "poses" / "gaze_views.csv"
    anchor_path = dataset_dir / "poses" / "focus_anchor.json"
    if not csv_path.exists():
        return False

    fields, rows = _read_gaze_rows(csv_path)
    if "view_bank_az_deg" not in fields or "view_bank_az_raw_deg" not in fields:
        return False
    if not rows:
        return False

    if anchor_path.exists():
        meta = json.loads(anchor_path.read_text(encoding="utf-8"))
        if meta.get("angle_method") not in (ANGLE_METHOD, None):
            return False
        bank_meta = meta.get("bank_meta") or {}
        if bank_meta and not bank_meta.get("colab_aligned", True):
            return False

    signed_err: list[float] = []
    for row in rows:
        bank = float(row.get("view_bank_az_deg", float("nan")))
        raw = float(row.get("view_bank_az_raw_deg", float("nan")))
        if _finite(bank) and _finite(raw):
            signed_err.append(abs(bank + raw))

    if len(signed_err) < 3:
        return False
    return float(np.median(signed_err)) <= ALIGN_TOL_DEG


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

    if not force and is_colab_aligned_gaze_export(dataset_dir):
        meta_path = dataset_dir / "poses" / "focus_anchor.json"
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["skipped"] = True
        meta["gaze_views_csv"] = str(csv_path)
        meta["angle_method"] = meta.get("angle_method", ANGLE_METHOD)
        meta["colab_aligned"] = True
        return meta

    if not force:
        print(
            "gaze_views.csv guncelleniyor (geometric + Colab bank az; "
            "eski/legacy CSV algilandi)..."
        )

    from export_gaze_views import export_gaze_views

    kwargs: dict = {
        "write_plot": write_plot,
        "angle_method": ANGLE_METHOD,
    }
    if gaze_threshold_deg is not None:
        kwargs["gaze_threshold_deg"] = float(gaze_threshold_deg)

    meta = export_gaze_views(dataset_dir, **kwargs)
    if not is_colab_aligned_gaze_export(dataset_dir):
        raise RuntimeError(
            "gaze_views.csv export sonrasi Colab hizali dogrulama basarisiz. "
            "Dataset'te depth + poses var mi kontrol edin."
        )
    meta["colab_aligned"] = True
    return meta


def strip_reference_mapping(
    dataset_dir: Path,
    *,
    targets: tuple[float, ...] = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0),
) -> list[dict]:
    """Same frame↔bank mapping as captures/export_angle_strip_15deg.py."""
    csv_path = Path(dataset_dir) / "poses" / "gaze_views.csv"
    _, rows = _read_gaze_rows(csv_path)
    bank = np.array([float(r["view_bank_az_deg"]) for r in rows], dtype=np.float64)
    out: list[dict] = []
    for target in targets:
        idx = int(np.argmin(np.abs(bank - float(target))))
        row = rows[idx]
        out.append(
            {
                "target_deg": float(target),
                "frame": int(float(row["frame"])),
                "view_bank_az_deg": float(row["view_bank_az_deg"]),
                "view_bank_az_raw_deg": float(row.get("view_bank_az_raw_deg", float("nan"))),
                "polyp_bank_az_deg": float(target),
            }
        )
    return out


def print_strip_reference_mapping(dataset_dir: Path) -> None:
    print("Strip referans eslestirme (view_bank_az_deg -> polyp bank):")
    for item in strip_reference_mapping(dataset_dir):
        print(
            f"  target {item['target_deg']:+5.0f}°  "
            f"Unity frame {item['frame']:4d}  "
            f"bank={item['view_bank_az_deg']:+.1f}°  "
            f"-> polyp view_az{'m' if item['polyp_bank_az_deg'] < 0 else 'p'}"
            f"{abs(int(round(item['polyp_bank_az_deg']))):03d}.png"
        )
