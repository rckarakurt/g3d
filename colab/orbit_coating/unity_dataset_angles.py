"""Unity dataset view-bank angles for Colab (geometric + Colab alignment).

Ensures ``poses/gaze_views.csv`` contains ``view_bank_az_deg`` from the
reference-free geometric export (``export_gaze_views --angle-method geometric``).

Polyp view-bank renders should use the same azimuths via
``unity_view_bank_angles.azimuths_from_gaze_dataset``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ANGLE_METHOD = "geometric"
GAZE_CSV = "poses/gaze_views.csv"
ANCHOR_JSON = "poses/focus_anchor.json"


def _read_gaze_csv(dataset_dir: Path) -> tuple[list[str], list[dict]]:
    csv_path = Path(dataset_dir) / GAZE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{GAZE_CSV} yok: {csv_path}\n"
            "Once Unity dataset yukleyin ve ensure_geometric_gaze_views() calistirin."
        )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def needs_geometric_export(dataset_dir: Path) -> bool:
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / GAZE_CSV
    anchor_path = dataset_dir / ANCHOR_JSON
    if not csv_path.exists():
        return True
    fieldnames, _ = _read_gaze_csv(dataset_dir)
    if "view_bank_az_deg" not in fieldnames:
        return True
    if not anchor_path.exists():
        return True
    meta = json.loads(anchor_path.read_text(encoding="utf-8"))
    return meta.get("angle_method") != ANGLE_METHOD


def validate_gaze_export(dataset_dir: Path) -> dict:
    """Raise if gaze_views.csv is not a geometric + Colab-aligned export."""
    dataset_dir = Path(dataset_dir)
    fieldnames, rows = _read_gaze_csv(dataset_dir)

    if "view_bank_az_deg" not in fieldnames:
        raise ValueError(
            "gaze_views.csv'de view_bank_az_deg yok. "
            "ensure_geometric_gaze_views(dataset, force=True) calistirin."
        )

    anchor_path = dataset_dir / ANCHOR_JSON
    anchor_meta: dict = {}
    if anchor_path.exists():
        anchor_meta = json.loads(anchor_path.read_text(encoding="utf-8"))
    if anchor_meta.get("angle_method") not in (ANGLE_METHOD, None):
        raise ValueError(
            f"focus_anchor.json angle_method={anchor_meta.get('angle_method')!r}; "
            f"beklenen {ANGLE_METHOD!r}. force=True ile yeniden export edin."
        )

    gazing_rows = [r for r in rows if int(r.get("is_gazing", 0)) == 1]
    use_rows = gazing_rows or rows
    bank = np.array(
        [float(r["view_bank_az_deg"]) for r in use_rows],
        dtype=np.float64,
    )
    bank = bank[np.isfinite(bank)]
    if len(bank) == 0:
        raise ValueError("view_bank_az_deg degerleri bos veya NaN.")

    raw_col = "view_bank_az_raw_deg" in fieldnames
    raw_span = None
    if raw_col:
        raw = np.array(
            [float(r.get("view_bank_az_raw_deg", float("nan"))) for r in use_rows],
            dtype=np.float64,
        )
        raw = raw[np.isfinite(raw)]
        if len(raw):
            raw_span = [float(np.min(raw)), float(np.max(raw))]

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "angle_method": anchor_meta.get("angle_method", ANGLE_METHOD),
        "frame_count": len(rows),
        "gazing_count": len(gazing_rows),
        "bank_az_min_deg": float(np.min(bank)),
        "bank_az_max_deg": float(np.max(bank)),
        "bank_az_raw_span_deg": raw_span,
        "mucosa_normal": anchor_meta.get("mucosa_normal"),
        "bank_meta": anchor_meta.get("bank_meta"),
        "convention": "view_bank_az_deg = -view_bank_az_raw_deg (Colab mesh_rotate_y)",
    }


def ensure_geometric_gaze_views(
    dataset_dir: Path,
    *,
    force: bool = False,
    write_plot: bool = False,
    **export_kwargs,
) -> dict:
    """Export or validate geometric gaze_views.csv with Colab-aligned bank azimuths."""
    from export_gaze_views import export_gaze_views

    dataset_dir = Path(dataset_dir)
    if force or needs_geometric_export(dataset_dir):
        if force:
            print("Geometric gaze export (force=True)...")
        else:
            print("Geometric gaze export (view_bank_az_deg)...")
        meta = export_gaze_views(
            dataset_dir,
            angle_method=ANGLE_METHOD,
            write_plot=write_plot,
            **export_kwargs,
        )
    else:
        meta = json.loads((dataset_dir / ANCHOR_JSON).read_text(encoding="utf-8"))

    summary = validate_gaze_export(dataset_dir)
    meta["bank_az_span_deg"] = [summary["bank_az_min_deg"], summary["bank_az_max_deg"]]
    meta["gaze_validation"] = summary
    return meta


def load_bank_azimuths(
    dataset_dir: Path,
    *,
    gazing_only: bool = True,
) -> list[float]:
    """Per-frame Colab-aligned bank azimuths from gaze_views.csv."""
    validate_gaze_export(dataset_dir)
    _, rows = _read_gaze_csv(dataset_dir)
    angles: list[float] = []
    for row in rows:
        if gazing_only and int(row.get("is_gazing", 0)) != 1:
            continue
        az = float(row.get("view_bank_az_deg", float("nan")))
        if np.isfinite(az):
            angles.append(az)
    return angles
