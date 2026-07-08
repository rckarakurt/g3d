"""Derive polyp view-bank azimuths from Unity gaze_views.csv (view_plane_deg)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from turntable_render import wall_azimuth_grid


def load_view_plane_angles(
    dataset_dir: Path,
    *,
    gazing_only: bool = True,
) -> list[float]:
    csv_path = Path(dataset_dir) / "poses" / "gaze_views.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"gaze_views.csv yok: {csv_path}\n"
            "Once Unity dataset + Bolum 6 (gaze export) calistirin."
        )
    angles: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if gazing_only and int(row.get("is_gazing", 0)) != 1:
                continue
            vp = float(row.get("view_plane_deg", float("nan")))
            if np.isfinite(vp):
                angles.append(vp)
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
    max_deg: float = 270.0,
    gazing_only: bool = True,
    fill_to_max: bool = True,
) -> tuple[list[float], dict]:
    """Unity view_plane_deg ile hizali view bank acilari (0..max_deg).

    - Unity'de gorulen acilar (bin'lenmis) her zaman dahil
    - fill_to_max=True: 0..max_deg arasi ayni adimla grid (bos acilar da render)
    """
    angles = load_view_plane_angles(dataset_dir, gazing_only=gazing_only)
    if not angles:
        step = int(bin_deg or 10)
        fallback = wall_azimuth_grid(step, int(max_deg))
        return fallback, {"source": "fallback_grid", "reason": "no_gaze_angles"}

    step = float(bin_deg if bin_deg is not None else suggest_bin_deg(angles))
    b = max(1, int(round(step)))

    unity_bins = sorted({float(int(round(a / b) * b)) for a in angles})
    lo = float(min(angles))
    hi_obs = float(max(angles))

    if fill_to_max:
        upper = int(max_deg)
        grid = [float(x) for x in range(0, upper + 1, b)]
        merged = sorted(set(grid) | set(unity_bins))
    else:
        upper = int(np.ceil(hi_obs / b) * b)
        merged = [float(x) for x in range(0, upper + 1, b)]

    meta = {
        "source": "unity_gaze_views",
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "gazing_only": gazing_only,
        "bin_deg": b,
        "max_deg": float(max_deg),
        "fill_to_max": fill_to_max,
        "unity_view_plane_min_deg": lo,
        "unity_view_plane_max_deg": hi_obs,
        "unity_unique_bins": len(unity_bins),
        "view_bank_count": len(merged),
        "view_bank_azimuths_deg": merged,
    }
    return merged, meta


def save_angle_meta(meta: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
