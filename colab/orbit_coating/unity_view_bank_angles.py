"""Derive polyp view-bank azimuths from Unity gaze_views.csv (view_bank_az_deg)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from turntable_render import (
    bank_az_to_unity_plane,
    wall_azimuth_grid,
)
from unity_dataset_angles import ensure_geometric_gaze_views, load_bank_azimuths


def load_view_plane_angles(
    dataset_dir: Path,
    *,
    gazing_only: bool = True,
    as_bank_az: bool = True,
    ensure_export: bool = True,
) -> list[float]:
    """Load per-frame bank azimuths (Colab-aligned, geometric export)."""
    dataset_dir = Path(dataset_dir)
    if ensure_export:
        ensure_geometric_gaze_views(dataset_dir, write_plot=False)
    angles = load_bank_azimuths(dataset_dir, gazing_only=gazing_only)
    if not as_bank_az:
        return [a + 90.0 for a in angles]
    return angles


def suggest_bin_deg(angles: list[float], *, default: float = 5.0) -> float:
    """Pick step similar to Unity orbit sampling."""
    if len(angles) < 2:
        return default
    arr = np.sort(np.asarray(angles, dtype=np.float64))
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0.05]
    if len(diffs) == 0:
        return default
    med = float(np.median(diffs))
    if med <= 3.0:
        return 5.0
    if med <= 7.0:
        return 5.0
    if med <= 12.0:
        return 10.0
    return 15.0


def azimuths_from_gaze_dataset(
    dataset_dir: Path,
    *,
    bin_deg: float | None = None,
    half_span_deg: float = 90.0,
    gazing_only: bool = True,
    fill_span: bool = True,
    ensure_export: bool = True,
) -> tuple[list[float], dict]:
    """View bank acilari: -half_span .. +half_span (0 = duz yuz, Y ekseni).

    Uses geometric ``view_bank_az_deg`` from Unity ``gaze_views.csv``
    (Colab-aligned: phi = -theta_geometric).
    """
    dataset_dir = Path(dataset_dir)
    if ensure_export:
        ensure_geometric_gaze_views(dataset_dir, write_plot=False)

    bank_angles = load_bank_azimuths(dataset_dir, gazing_only=gazing_only)
    span = min(90.0, float(half_span_deg))
    if not bank_angles:
        step = int(bin_deg or 5)
        fallback = wall_azimuth_grid(step, int(span))
        return fallback, {"source": "fallback_grid", "reason": "no_gaze_angles"}

    step = float(bin_deg if bin_deg is not None else suggest_bin_deg(bank_angles))
    b = max(1, int(round(step)))

    unity_bins = sorted({float(int(round(a / b) * b)) for a in bank_angles})
    lo = float(min(bank_angles))
    hi_obs = float(max(bank_angles))

    if fill_span:
        upper = int(span)
        grid = [float(x) for x in range(-upper, upper + 1, b)]
        merged = sorted(set(grid) | set(unity_bins))
    else:
        lo_b = int(np.floor(lo / b) * b)
        hi_b = int(np.ceil(hi_obs / b) * b)
        merged = [float(x) for x in range(lo_b, hi_b + 1, b)]

    meta = {
        "source": "unity_gaze_views",
        "dataset_dir": str(dataset_dir.resolve()),
        "gazing_only": gazing_only,
        "bin_deg": b,
        "half_span_deg": span,
        "fill_span": fill_span,
        "angle_method": "geometric",
        "convention": (
            "view_bank_az_deg: geometric orbit az, Colab-aligned (mesh_rotate_y -az)"
        ),
        "uses_view_bank_az_deg": True,
        "bank_az_min_deg": lo,
        "bank_az_max_deg": hi_obs,
        "unity_plane_min_deg": bank_az_to_unity_plane(lo),
        "unity_plane_max_deg": bank_az_to_unity_plane(hi_obs),
        "unity_unique_bins": len(unity_bins),
        "view_bank_count": len(merged),
        "view_bank_azimuths_deg": merged,
    }
    return merged, meta


def save_angle_meta(meta: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
