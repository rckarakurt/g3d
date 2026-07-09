"""Paper figures: 7 fixed Unity frame + textured polyp view-bank composites.

Each pair uses an explicit frame index and bank azimuth (15° steps -45..+45),
matching export_angle_strip_15deg.py frame selection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Unity frame index -> Colab view-bank azimuth (degrees)
PAPER_ANGLE_PAIRS: tuple[tuple[int, float], ...] = (
    (67, -45.0),
    (78, -30.0),
    (87, -15.0),
    (98, 0.0),
    (121, 15.0),
    (132, 30.0),
    (143, 45.0),
)


def _load_gaze_by_frame(dataset_dir: Path) -> dict[int, dict]:
    path = dataset_dir / "poses" / "gaze_views.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(float(r["frame"])): r for r in rows}


def _load_bank_index(view_bank_dir: Path) -> tuple[dict, dict[float, Path], float]:
    from gaze_composite import resolve_polyp_diameter_mm

    manifest_path = view_bank_dir / "view_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"View bank manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    polyp_diameter_mm = resolve_polyp_diameter_mm(manifest, view_bank_dir)
    bank_files = {
        float(v["azimuth_deg"]): view_bank_dir / v["file"] for v in manifest["views"]
    }
    for az, path in bank_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing view bank image for az={az}: {path}")
    return manifest, bank_files, polyp_diameter_mm


def _resolve_polyp_path(bank_files: dict[float, Path], az_deg: float) -> Path:
    if float(az_deg) in bank_files:
        return bank_files[float(az_deg)]
    nearest = min(bank_files.keys(), key=lambda k: abs(k - float(az_deg)))
    if abs(nearest - float(az_deg)) > 0.5:
        raise FileNotFoundError(
            f"View bank missing azimuth {az_deg:+.0f}° (nearest {nearest:+.0f}°)"
        )
    return bank_files[nearest]


def export_paper_angle_composites(
    dataset_dir: Path,
    view_bank_dir: Path,
    out_dir: Path,
    *,
    pairs: tuple[tuple[int, float], ...] = PAPER_ANGLE_PAIRS,
    scale_boost: float = 8.0,
    lab_match: bool = True,
    strip_cell_h: int = 360,
    strip_label_h: int = 56,
) -> dict:
    """Composite textured polyp onto Unity mucosa center; save singles + strip."""
    from gaze_composite import (
        estimate_target_diameter_px,
        load_camera,
        load_rgba,
        paste_rgba_center,
        resize_patch_rgba,
    )

    dataset_dir = Path(dataset_dir).resolve()
    view_bank_dir = Path(view_bank_dir).resolve()
    out_dir = Path(out_dir).resolve()
    singles_dir = out_dir / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)

    _, bank_files, polyp_diameter_mm = _load_bank_index(view_bank_dir)
    cam = load_camera(dataset_dir)
    fx = float(cam["fx"])
    cx_default = float(cam["cx"])
    cy_default = float(cam["cy"])
    gaze_by_frame = _load_gaze_by_frame(dataset_dir)

    distances = []
    for frame, _ in pairs:
        row = gaze_by_frame.get(frame)
        if row is not None:
            d = float(row.get("distance_m", float("nan")))
            if np.isfinite(d):
                distances.append(d)
    global_distance_m = float(np.median(distances)) if distances else float("nan")

    results: list[dict] = []
    strip_panels: list[np.ndarray] = []

    for frame, bank_az in pairs:
        rgb_path = dataset_dir / "rgb" / f"{frame:06d}.png"
        if not rgb_path.exists():
            raise FileNotFoundError(f"Unity RGB missing: {rgb_path}")

        unity_bgr = cv2.imread(str(rgb_path))
        if unity_bgr is None:
            raise FileNotFoundError(f"Cannot read Unity RGB: {rgb_path}")
        unity_rgb = cv2.cvtColor(unity_bgr, cv2.COLOR_BGR2RGB)

        row = gaze_by_frame.get(frame, {})
        anchor_u = float(row.get("anchor_u", cx_default))
        anchor_v = float(row.get("anchor_v", cy_default))
        distance_m = float(row.get("distance_m", global_distance_m))
        if not np.isfinite(distance_m):
            distance_m = global_distance_m

        polyp_path = _resolve_polyp_path(bank_files, bank_az)
        patch_rgba = load_rgba(polyp_path)
        target_d = estimate_target_diameter_px(
            distance_m,
            fx,
            polyp_diameter_mm,
            scale_boost=scale_boost,
        )
        patch_scaled = resize_patch_rgba(patch_rgba, target_d)
        composite_rgb, _ = paste_rgba_center(
            unity_rgb,
            patch_scaled,
            anchor_u,
            anchor_v,
            lab_match=lab_match,
        )

        sign = "m" if bank_az < 0 else "p"
        az_label = f"{sign}{abs(int(round(bank_az))):03d}"
        stem = f"composite_f{frame:04d}_az{az_label}"
        single_path = singles_dir / f"{stem}.png"
        cv2.imwrite(str(single_path), cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))

        measured_bank = float(row.get("view_bank_az_deg", float("nan")))
        entry = {
            "frame": frame,
            "image_name": f"image_{frame:04d}.png",
            "target_bank_az_deg": float(bank_az),
            "measured_view_bank_az_deg": measured_bank,
            "polyp_view_file": polyp_path.name,
            "anchor_u": anchor_u,
            "anchor_v": anchor_v,
            "distance_m": distance_m,
            "target_diameter_px": target_d,
            "composite": str(single_path.relative_to(out_dir)),
        }
        results.append(entry)
        print(
            f"  frame {frame:4d} ({entry['image_name']})  "
            f"bank {bank_az:+.0f}°  -> {polyp_path.name}"
        )

        # Strip panel: label + composite thumbnail
        h, w = composite_rgb.shape[:2]
        scale = strip_cell_h / max(h, 1)
        thumb = cv2.resize(
            composite_rgb,
            (max(1, int(w * scale)), strip_cell_h),
            interpolation=cv2.INTER_AREA,
        )
        label_bar = np.zeros((strip_label_h, thumb.shape[1], 3), dtype=np.uint8)
        lines = [
            f"image_{frame:04d}  ({bank_az:+.0f} deg)",
            f"polyp {polyp_path.name}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                label_bar,
                line,
                (8, 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 240),
                1,
                cv2.LINE_AA,
            )
        strip_panels.append(np.vstack([label_bar, thumb]))

    gap = 8
    strip = strip_panels[0]
    for panel in strip_panels[1:]:
        g = np.full((strip.shape[0], gap, 3), 24, dtype=np.uint8)
        strip = np.hstack([strip, g, panel])

    strip_path = out_dir / "paper_composites_strip_7.png"
    cv2.imwrite(str(strip_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    manifest_path = out_dir / "paper_composites_manifest.json"
    summary = {
        "dataset_dir": str(dataset_dir),
        "view_bank_dir": str(view_bank_dir),
        "out_dir": str(out_dir),
        "pairs": [{"frame": f, "bank_az_deg": a} for f, a in pairs],
        "count": len(results),
        "strip": str(strip_path.name),
        "singles_dir": singles_dir.name,
        "entries": results,
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
