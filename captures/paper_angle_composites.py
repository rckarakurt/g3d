"""Paper figures: 7 fixed Unity frame + textured polyp view-bank composites.

Polyp is pasted onto Unity mucosa center (single overlaid image per pair).
Outputs: 7 full-resolution composites + one horizontal strip for publication.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from eccv_figure import build_full_width_row, save_eccv_figure

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

# Endoscopic frames are slightly wider than tall in Unity export.
ECCV_COMPOSITE_ASPECT = 0.75


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
    scale_boost: float = 16.0,
    lab_match: bool = True,
    strip_aspect: float = ECCV_COMPOSITE_ASPECT,
) -> dict:
    """Paste textured polyp onto Unity mucosa center; save 7 overlays + ECCV strip."""
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
    strip_images: list[np.ndarray] = []
    strip_labels: list[str] = []
    strip_subtitles: list[str] = []

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

        panel_idx = len(strip_images)
        strip_images.append(composite_rgb)
        strip_labels.append(f"({chr(ord('a') + panel_idx)})")
        strip_subtitles.append(f"{bank_az:+.0f}°")

        measured_bank = float(row.get("view_bank_az_deg", float("nan")))
        results.append(
            {
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
        )
        print(
            f"  frame {frame:4d}  bank {bank_az:+.0f}°  "
            f"-> {polyp_path.name}  (polyp mucosa ortasina bindirildi)"
        )

    strip_rgb = build_full_width_row(
        strip_images,
        strip_labels,
        aspect=strip_aspect,
        subtitles=strip_subtitles,
    )
    figure_paths = save_eccv_figure(
        strip_rgb,
        out_dir / "paper_composites_strip_7.png",
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "view_bank_dir": str(view_bank_dir),
        "out_dir": str(out_dir),
        "pairs": [{"frame": f, "bank_az_deg": a} for f, a in pairs],
        "count": len(results),
        "composites_strip": Path(figure_paths["png"]).name if figure_paths.get("png") else None,
        "composites_strip_path": figure_paths.get("png"),
        "composites_strip_pdf": figure_paths.get("pdf"),
        "eccv_textwidth_mm": 122.0,
        "latex_caption_hint": (
            "\\caption{Trajectory-indexed composites at seven matched viewing angles. "
            "Panels (a--g): $\\psi=-45^\\circ,\\ldots,+45^\\circ$. "
            "Each image overlays the same textured lesion mesh on simulated mucosa.}"
        ),
        "singles_dir": singles_dir.name,
        "description": "Polyp pasted on Unity mucosa center; no side-by-side Unity/polyp panels",
        "entries": results,
    }
    (out_dir / "paper_composites_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    latex_path = out_dir / "paper_angle_pairs_table.tex"
    latex_path.write_text(
        export_paper_pairs_latex_table(results),
        encoding="utf-8",
    )
    summary["angle_pairs_latex"] = str(latex_path)
    return summary


def export_paper_pairs_latex_table(
    entries: list[dict],
    *,
    caption: str = "Unity frame to view-bank azimuth mapping.",
) -> str:
    """ECCV booktabs table for the seven fixed angle pairs."""
    body = "\n".join(
        f"    {e['image_name']} & {e['target_bank_az_deg']:+.0f}$^\\circ$ & "
        f"{e['polyp_view_file']} \\\\"
        for e in entries
    )
    return (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{tab:paper-angle-pairs}\n"
        "  \\begin{tabular}{lcl}\n"
        "    \\toprule\n"
        "    Unity frame & Bank az. & View-bank file \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
