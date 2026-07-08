"""Save RGB frames with computed anchor point and in-plane viewing angle overlay.

Usage:
  python export_gaze_rgb_marked.py --dataset captures/medical_gan_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRNAME = "gaze_rgb"


def save_rgb_png(path: Path, bgr: np.ndarray) -> None:
    """Write a true RGB-order PNG (not OpenCV BGR on disk)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path, format="PNG")
    except ImportError:
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def load_gaze_rows(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "poses" / "gaze_views.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run export_gaze_views.py first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plane_angle_from_row(row: dict) -> float:
    if row.get("view_plane_deg"):
        value = float(row["view_plane_deg"])
        if np.isfinite(value):
            return value
    az = float(row.get("view_azimuth_deg", float("nan")))
    if not np.isfinite(az):
        return float("nan")
    az360 = (az + 360.0) % 360.0
    return az360 if az360 <= 180.0 else 360.0 - az360


def plane_tilt_from_row(row: dict) -> float:
    if row.get("view_tilt_deg"):
        value = float(row["view_tilt_deg"])
        if np.isfinite(value):
            return value
    return float(row.get("view_elevation_deg", float("nan")))


def draw_plane_gauge(
    img: np.ndarray,
    *,
    center: tuple[int, int],
    radius: int,
    angle_deg: float,
) -> None:
    """Semicircle gauge 0..180 deg on anchor tangent plane."""
    cx, cy = center
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, 180, 360, (220, 220, 220), 2, cv2.LINE_AA)
    for tick in (0, 45, 90, 135, 180):
        rad = math.radians(180.0 - tick)
        x0 = int(cx + (radius - 6) * math.cos(rad))
        y0 = int(cy - (radius - 6) * math.sin(rad))
        x1 = int(cx + radius * math.cos(rad))
        y1 = int(cy - radius * math.sin(rad))
        cv2.line(img, (x0, y0), (x1, y1), (180, 180, 180), 2, cv2.LINE_AA)
        label = f"{tick}"
        tx = int(cx + (radius + 14) * math.cos(rad)) - 8
        ty = int(cy - (radius + 14) * math.sin(rad)) + 5
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    if np.isfinite(angle_deg):
        needle = max(0.0, min(180.0, float(angle_deg)))
        rad = math.radians(180.0 - needle)
        nx = int(cx + (radius - 4) * math.cos(rad))
        ny = int(cy - (radius - 4) * math.sin(rad))
        cv2.line(img, (cx, cy), (nx, ny), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, (0, 220, 255), -1, cv2.LINE_AA)


def draw_text_block(img: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (16, 16, 16), 1, cv2.LINE_AA)
        y += 24


def draw_anchor_overlay(
    bgr: np.ndarray,
    u: float,
    v: float,
    *,
    frame: int,
    plane_deg: float,
    tilt_deg: float,
    gaze_error: float,
    reproj_error_px: float,
    is_gazing: bool,
    center_u: float,
    center_v: float,
    anchor_disk_radius_px: float | None = None,
    azimuth_deg: float | None = None,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    ax, ay = int(round(u)), int(round(v))
    cx, cy = int(round(center_u)), int(round(center_v))

    anchor_color = (0, 255, 0) if is_gazing else (0, 180, 255)
    cv2.drawMarker(out, (ax, ay), anchor_color, cv2.MARKER_TILTED_CROSS, 28, 2)
    if anchor_disk_radius_px is not None and np.isfinite(anchor_disk_radius_px) and anchor_disk_radius_px > 2:
        cv2.circle(out, (ax, ay), int(round(anchor_disk_radius_px)), anchor_color, 2)
    cv2.circle(out, (ax, ay), 10, anchor_color, 2)
    cv2.circle(out, (ax, ay), 3, anchor_color, -1)

    gauge_cx = w - 95
    gauge_cy = h - 28
    draw_plane_gauge(out, center=(gauge_cx, gauge_cy), radius=70, angle_deg=plane_deg)

    plane_text = f"{plane_deg:.1f}" if np.isfinite(plane_deg) else "?"
    tilt_text = f"{tilt_deg:.1f}" if np.isfinite(tilt_deg) else "?"
    lines = [
        f"frame {frame:03d}",
        f"plane: {plane_text} deg  (0-180)",
        f"tilt: {tilt_text} deg",
    ]
    if azimuth_deg is not None and np.isfinite(azimuth_deg):
        lines.append(f"azimuth: {azimuth_deg:.1f} deg")
    lines.extend(
        [
            f"ray sapma: {gaze_error:.1f} deg",
            f"anchor: merkez ({ax}, {ay})",
            "yesil = anchor (crosshair)",
        ]
    )
    draw_text_block(out, lines, (12, 28))

    cv2.putText(
        out,
        f"{plane_text}",
        (gauge_cx - 28, gauge_cy - 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "deg",
        (gauge_cx + 8, gauge_cy - 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def export_marked_rgb(
    dataset_dir: Path,
    output_dir: Path | None = None,
    *,
    gazing_only: bool = True,
    clean: bool = True,
) -> Path:
    dataset_dir = dataset_dir.resolve()
    rows = load_gaze_rows(dataset_dir)
    meta = json.loads((dataset_dir / "poses" / "focus_anchor.json").read_text(encoding="utf-8"))
    center_u, center_v = meta["principal_point"]
    patch_r = float(meta.get("anchor_patch_radius_px", 20))
    spatial_r = float(meta.get("anchor_spatial_radius_m", meta.get("anchor_inlier_radius_m", 0.12)))

    def anchor_disk_px(_distance_m: float) -> float:
        return patch_r

    if gazing_only:
        selected = [r for r in rows if int(r.get("is_gazing", 0)) == 1]
    else:
        selected = rows

    if not selected:
        raise RuntimeError("No frames selected. Check gaze_views.csv / is_gazing.")

    base_dir = output_dir or (dataset_dir / DEFAULT_OUTPUT_DIRNAME)
    rgb_out_dir = base_dir / "rgb"
    if clean and base_dir.exists():
        shutil.rmtree(base_dir)
    rgb_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    src_rgb_dir = dataset_dir / "rgb"

    for row in selected:
        frame = int(float(row["frame"]))
        rgb_path = src_rgb_dir / f"{frame:06d}.png"
        if not rgb_path.exists():
            rgb_path = dataset_dir.parent / "image" / f"image_{frame:04d}.png"
        if not rgb_path.exists():
            continue

        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            continue

        u = float(row["anchor_u"])
        v = float(row["anchor_v"])
        plane_deg = plane_angle_from_row(row)
        tilt_deg = plane_tilt_from_row(row)
        err = float(row.get("ray_dev_deg", row.get("gaze_error_deg", 0)))
        reproj = float(row.get("reproj_error_px", float("nan")))
        is_gazing = int(row.get("is_gazing", 0)) == 1

        marked = draw_anchor_overlay(
            bgr,
            center_u,
            center_v,
            frame=frame,
            plane_deg=plane_deg,
            tilt_deg=tilt_deg,
            gaze_error=err,
            reproj_error_px=reproj,
            is_gazing=is_gazing,
            center_u=center_u,
            center_v=center_v,
            anchor_disk_radius_px=anchor_disk_px(0.0),
        )

        plane_i = int(round(plane_deg)) if np.isfinite(plane_deg) else 0
        out_name = f"{frame:06d}.png"
        out_path = rgb_out_dir / out_name
        save_rgb_png(out_path, marked)

        manifest_rows.append(
            {
                "frame": frame,
                "rgb": f"rgb/{out_name}",
                "output": out_name,
                "label": f"frame_{frame:06d}_plane{plane_i:03d}",
                "anchor_u": u,
                "anchor_v": v,
                "view_plane_deg": plane_deg,
                "view_tilt_deg": tilt_deg,
                "view_azimuth_deg": float(row.get("view_azimuth_deg", float("nan"))),
                "view_elevation_deg": float(row.get("view_elevation_deg", float("nan"))),
                "gaze_error_deg": err,
                "reproj_error_px": reproj,
                "is_gazing": int(is_gazing),
            }
        )

    manifest_csv = base_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = base_dir / "README.txt"
    readme.write_text(
        f"""Gaze RGB export (anchor + duzlem acisi)
==========================================
Dataset: {dataset_dir.name}
Gazing only: {gazing_only}
Frames exported: {len(manifest_rows)}
Gaze threshold (deg): {meta.get('gaze_threshold_deg', '?')}

Klasor yapisi
-------------
rgb/000053.png   — isaretli RGB kareler (gercek RGB PNG)
manifest.csv     — frame, rgb yolu, acilar, anchor pikseli

Aciklama
--------
view_plane_deg (0-180): anchor duzleminde bakis acisi
view_tilt_deg: duzleme gore egim

Gorsel
------
Yesil X  : hesaplanan anchor (project 3D)
Sari +   : image center (referans)
Sag alt  : 0-180 derece gostergesi
""",
        encoding="utf-8",
    )

    print(f"Saved {len(manifest_rows)} RGB PNG -> {rgb_out_dir}")
    return rgb_out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RGB with computed anchor and plane angle.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "medical_gan_dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Default: <dataset>/{DEFAULT_OUTPUT_DIRNAME}/rgb/",
    )
    parser.add_argument("--all-frames", action="store_true", help="Include non-gazing frames too")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    export_marked_rgb(
        args.dataset,
        args.output,
        gazing_only=not args.all_frames,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
