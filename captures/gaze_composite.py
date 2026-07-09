"""Trajectory-consistent polyp compositing onto Unity RGB frames.

Smooths view angles / scale along the orbit, blends adjacent view-bank renders
for sub-bin continuity, and pastes at the gaze anchor (screen center).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def _resolve_orbit_scripts() -> Path:
    """Local captures/ vs Colab GitHub clone / legacy embed."""
    for candidate in (
        ROOT.parent / "colab" / "orbit_coating",
        ROOT.parent,
        Path("/content/g3d/colab/orbit_coating"),
        Path("/content/vrcaps_scripts"),
    ):
        if (candidate / "coating_utils.py").exists():
            return candidate
    return ROOT.parent / "colab" / "orbit_coating"


ORBIT = _resolve_orbit_scripts()
sys.path.insert(0, str(ORBIT))

from coating_utils import export_trajectory_mp4, lab_color_match_polyp  # noqa: E402
from turntable_render import unity_plane_to_bank_az  # noqa: E402


def load_rgba(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        gray = img
        return np.dstack([gray, gray, gray, np.full_like(gray, 255)])
    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    bgra = img
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    return np.dstack([rgb, bgra[:, :, 3]])


def polyp_mask_bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def estimate_target_diameter_px(
    distance_m: float,
    fx: float,
    polyp_diameter_mm: float,
    *,
    scale_boost: float = 1.0,
    min_px: float = 24.0,
    max_px: float = 220.0,
) -> float:
    if not np.isfinite(distance_m) or distance_m <= 1e-4:
        return min_px
    diameter_m = polyp_diameter_mm / 1000.0
    px = fx * diameter_m / distance_m * scale_boost
    return float(np.clip(px, min_px, max_px))


def resize_patch_rgba(rgba: np.ndarray, target_diameter_px: float) -> np.ndarray:
    alpha = rgba[:, :, 3]
    bbox = polyp_mask_bbox(alpha)
    if bbox is None:
        return rgba
    x0, y0, x1, y1 = bbox
    crop = rgba[y0:y1, x0:x1]
    cur_d = max(x1 - x0, y1 - y0, 1)
    scale = target_diameter_px / cur_d
    new_w = max(4, int(round(crop.shape[1] * scale)))
    new_h = max(4, int(round(crop.shape[0] * scale)))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)


def smooth_series(values: np.ndarray, window: int, *, method: str = "median") -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(np.float64)
    try:
        from scipy.ndimage import median_filter, uniform_filter1d

        if method == "median":
            return median_filter(values.astype(np.float64), size=window, mode="nearest")
        return uniform_filter1d(values.astype(np.float64), size=window, mode="nearest")
    except ImportError:
        # Fallback: simple moving average
        out = values.astype(np.float64).copy()
        half = window // 2
        for i in range(len(out)):
            lo = max(0, i - half)
            hi = min(len(out), i + half + 1)
            out[i] = float(np.mean(values[lo:hi]))
        return out


def blend_azimuth_pair(
    view_plane_deg: float,
    bank_az: list[float],
) -> tuple[float, float, float]:
    """Return (az_lo, az_hi, weight_hi) for linear bank blending."""
    bank = sorted(float(a) for a in bank_az)
    theta = float(view_plane_deg)
    if not np.isfinite(theta) or not bank:
        return bank[0], bank[0], 0.0
    if theta <= bank[0]:
        return bank[0], bank[0], 0.0
    if theta >= bank[-1]:
        return bank[-1], bank[-1], 0.0
    for lo, hi in zip(bank[:-1], bank[1:]):
        if lo <= theta <= hi:
            span = hi - lo
            w_hi = (theta - lo) / span if span > 1e-6 else 0.0
            return lo, hi, float(np.clip(w_hi, 0.0, 1.0))
    nearest = min(bank, key=lambda az: abs(az - theta))
    return nearest, nearest, 0.0


def resize_to_common(rgba_a: np.ndarray, rgba_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ha, wa = rgba_a.shape[:2]
    hb, wb = rgba_b.shape[:2]
    h = max(ha, hb)
    w = max(wa, wb)
    if (ha, wa) != (h, w):
        rgba_a = cv2.resize(rgba_a, (w, h), interpolation=cv2.INTER_LINEAR)
    if (hb, wb) != (h, w):
        rgba_b = cv2.resize(rgba_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgba_a, rgba_b


def blend_rgba_patches(rgba_lo: np.ndarray, rgba_hi: np.ndarray, weight_hi: float) -> np.ndarray:
    if weight_hi <= 1e-6:
        return rgba_lo
    if weight_hi >= 1.0 - 1e-6:
        return rgba_hi
    lo, hi = resize_to_common(rgba_lo, rgba_hi)
    w = float(weight_hi)
    rgb = lo[:, :, :3].astype(np.float32) * (1.0 - w) + hi[:, :, :3].astype(np.float32) * w
    alpha = lo[:, :, 3].astype(np.float32) * (1.0 - w) + hi[:, :, 3].astype(np.float32) * w
    return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), np.clip(alpha, 0, 255).astype(np.uint8)])


def build_interpolated_patch(
    view_plane_deg: float,
    bank_az: list[float],
    bank_files: dict[float, Path],
    target_diameter_px: float,
    *,
    cache: dict[float, np.ndarray],
) -> tuple[np.ndarray, float, float, float]:
    az_lo, az_hi, w_hi = blend_azimuth_pair(view_plane_deg, bank_az)

    def _scaled(az: float) -> np.ndarray:
        if az not in cache:
            cache[az] = load_rgba(bank_files[az])
        return resize_patch_rgba(cache[az], target_diameter_px)

    if az_lo == az_hi:
        return _scaled(az_lo), az_lo, az_hi, w_hi
    patch = blend_rgba_patches(_scaled(az_lo), _scaled(az_hi), w_hi)
    return patch, az_lo, az_hi, w_hi


def paste_rgba_center(
    base_rgb: np.ndarray,
    patch_rgba: np.ndarray,
    center_u: float,
    center_v: float,
    *,
    lab_match: bool = True,
    lab_strength: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = base_rgb.shape[:2]
    ph, pw = patch_rgba.shape[:2]
    cx = int(round(center_u))
    cy = int(round(center_v))
    x0 = cx - pw // 2
    y0 = cy - ph // 2
    x1 = x0 + pw
    y1 = y0 + ph

    out = base_rgb.copy()
    paste_mask = np.zeros((h, w), dtype=np.uint8)

    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(w, x1)
    dy1 = min(h, y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return out, paste_mask

    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)
    patch = patch_rgba[sy0:sy1, sx0:sx1]
    region = out[dy0:dy1, dx0:dx1].astype(np.float32)
    rgb = patch[:, :, :3].astype(np.float32)
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0

    if lab_match and alpha.max() > 0:
        mask_u8 = (alpha[:, :, 0] > 0.05).astype(np.uint8) * 255
        rgb = lab_color_match_polyp(
            rgb.astype(np.uint8),
            region.astype(np.uint8),
            mask_u8,
            strength=lab_strength,
        ).astype(np.float32)

    blended = region * (1.0 - alpha) + rgb * alpha
    out[dy0:dy1, dx0:dx1] = np.clip(blended, 0, 255).astype(np.uint8)
    paste_mask[dy0:dy1, dx0:dx1] = np.where(alpha[:, :, 0] > 0.05, 255, 0).astype(np.uint8)
    return out, paste_mask


def _draw_label_bar(img: np.ndarray, text: str, *, bar_h: int = 32) -> np.ndarray:
    out = img.copy()
    bar = np.zeros((bar_h, out.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, out])


def _patch_on_background(patch_rgba: np.ndarray, bg: tuple[int, int, int] = (12, 12, 16)) -> np.ndarray:
    rgb = patch_rgba[:, :, :3].astype(np.float32)
    a = patch_rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg_arr = np.array(bg, dtype=np.float32)
    out = rgb * a + bg_arr * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_debug_triptych(
    out_path: Path,
    *,
    unity_rgb: np.ndarray,
    patch_rgba: np.ndarray,
    composite_rgb: np.ndarray,
    view_plane_deg: float,
    bank_az_lo: float,
    bank_az_hi: float,
    blend_w_hi: float,
    frame: int,
) -> None:
    """Unity | mesh (matched angle) | composite — aci etiketli."""
    thumb_h = 280
    gap = np.full((thumb_h + 32, 8, 3), 20, dtype=np.uint8)

    def _prep(img: np.ndarray, label: str) -> np.ndarray:
        h, w = img.shape[:2]
        scale = thumb_h / max(h, 1)
        thumb = cv2.resize(img, (max(1, int(w * scale)), thumb_h), interpolation=cv2.INTER_AREA)
        return _draw_label_bar(thumb, label)

    patch_rgb = _patch_on_background(patch_rgba)
    az_txt = (
        f"mesh az {bank_az_lo:.0f}-{bank_az_hi:.0f} (w={blend_w_hi:.2f})"
        if bank_az_lo != bank_az_hi
        else f"mesh az {bank_az_lo:.0f}"
    )
    panels = [
        _prep(unity_rgb, f"Unity f{frame:04d}  bank_az={view_plane_deg:+.1f}"),
        _prep(patch_rgb, az_txt),
        _prep(composite_rgb, "composite"),
    ]
    strip = panels[0]
    for p in panels[1:]:
        strip = np.hstack([strip, gap[: strip.shape[0]], p])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def resolve_polyp_diameter_mm(bank_manifest: dict, view_bank_dir: Path) -> float:
    if "polyp_diameter_mm" in bank_manifest:
        return float(bank_manifest["polyp_diameter_mm"])
    meta_path = view_bank_dir.parent / "pipeline_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        radius = float(meta.get("mesh_radius_mm", 6.0))
        return radius * 2.0
    return 12.0


def load_gaze_rows(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "poses" / "gaze_views.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run export_gaze_views.py first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_camera(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "camera.json").read_text(encoding="utf-8"))


def prepare_trajectory_rows(
    gaze_rows: list[dict],
    *,
    gazing_only: bool,
    smooth_window: int,
    smooth_angles: bool,
    smooth_distance: bool,
    smooth_anchor: bool,
) -> list[dict]:
    rows: list[dict] = []
    for row in gaze_rows:
        if gazing_only and int(row.get("is_gazing", 0)) != 1:
            continue
        rgb_path = None  # filled later
        rows.append(dict(row))

    if not rows:
        return rows

    rows.sort(key=lambda r: int(float(r["frame"])))
    n = len(rows)

    def col(name: str, default: float = float("nan")) -> np.ndarray:
        vals = []
        for r in rows:
            try:
                v = float(r.get(name, default))
            except (TypeError, ValueError):
                v = default
            vals.append(v)
        return np.asarray(vals, dtype=np.float64)

    if smooth_angles and smooth_window > 1:
        angle_col = "view_bank_az_deg" if "view_bank_az_deg" in rows[0] else "view_plane_deg"
        vp = col(angle_col)
        if np.any(np.isfinite(vp)):
            vp_smooth = smooth_series(np.nan_to_num(vp, nan=np.nanmedian(vp)), smooth_window)
            for i, r in enumerate(rows):
                r[f"{angle_col}_raw"] = r.get(angle_col, "")
                r[angle_col] = str(vp_smooth[i])

    if smooth_distance and smooth_window > 1:
        dist = col("distance_m")
        if np.any(np.isfinite(dist)):
            dist_smooth = smooth_series(np.nan_to_num(dist, nan=np.nanmedian(dist)), smooth_window)
            for i, r in enumerate(rows):
                r["distance_m_raw"] = r.get("distance_m", "")
                r["distance_m"] = str(dist_smooth[i])

    if smooth_anchor and smooth_window > 1:
        for axis in ("anchor_u", "anchor_v"):
            arr = col(axis)
            if np.any(np.isfinite(arr)):
                sm = smooth_series(np.nan_to_num(arr, nan=np.nanmedian(arr)), smooth_window)
                for i, r in enumerate(rows):
                    r[f"{axis}_raw"] = r.get(axis, "")
                    r[axis] = str(sm[i])

    return rows


def composite_dataset(
    dataset_dir: Path,
    view_bank_dir: Path,
    out_dir: Path,
    *,
    gazing_only: bool = True,
    scale_boost: float = 8.0,
    global_scale: bool = True,
    smooth_window: int = 7,
    smooth_angles: bool = True,
    smooth_distance: bool = True,
    smooth_anchor: bool = False,
    bank_blend: bool = True,
    lab_match: bool = True,
    max_frames: int | None = None,
    write_mp4: bool = True,
    write_debug: bool = False,
    debug_max_frames: int = 12,
    mp4_fps: float = 15.0,
) -> dict:
    dataset_dir = dataset_dir.resolve()
    view_bank_dir = view_bank_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_out = out_dir / "rgb"
    rgb_out.mkdir(parents=True, exist_ok=True)

    manifest_path = view_bank_dir / "view_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"View bank manifest missing: {manifest_path}")
    bank_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    polyp_diameter_mm = resolve_polyp_diameter_mm(bank_manifest, view_bank_dir)

    bank_az = sorted(float(v["azimuth_deg"]) for v in bank_manifest["views"])
    bank_files = {float(v["azimuth_deg"]): view_bank_dir / v["file"] for v in bank_manifest["views"]}
    for az, path in bank_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing view bank image: {path}")

    cam = load_camera(dataset_dir)
    fx = float(cam["fx"])
    cx_default = float(cam["cx"])
    cy_default = float(cam["cy"])

    traj_rows = prepare_trajectory_rows(
        load_gaze_rows(dataset_dir),
        gazing_only=gazing_only,
        smooth_window=smooth_window,
        smooth_angles=smooth_angles,
        smooth_distance=smooth_distance,
        smooth_anchor=smooth_anchor,
    )

    if global_scale and traj_rows:
        dists = [float(r["distance_m"]) for r in traj_rows if np.isfinite(float(r.get("distance_m", float("nan"))))]
        global_distance_m = float(np.median(dists)) if dists else float("nan")
    else:
        global_distance_m = float("nan")

    debug_dir = out_dir / "debug"
    if write_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Angle match: {len(bank_az)} bank views, span {bank_az[0]:.0f}..{bank_az[-1]:.0f}")
    if traj_rows:
        if "view_bank_az_deg" in traj_rows[0]:
            vp0 = float(traj_rows[0].get("view_bank_az_deg", 0))
            vp1 = float(traj_rows[-1].get("view_bank_az_deg", 0))
            print(f"Unity trajectory view_bank_az: {vp0:.1f} .. {vp1:.1f}  ({len(traj_rows)} frames)")
        else:
            vp0 = float(traj_rows[0].get("view_plane_deg", 0))
            vp1 = float(traj_rows[-1].get("view_plane_deg", 0))
            print(f"Unity trajectory view_plane (legacy): {vp0:.1f} .. {vp1:.1f}  ({len(traj_rows)} frames)")

    manifest_rows: list[dict] = []
    rgba_cache: dict[float, np.ndarray] = {}
    composite_paths: list[Path] = []
    count = 0

    for row in traj_rows:
        if max_frames is not None and count >= max_frames:
            break
        frame = int(float(row["frame"]))
        rgb_path = dataset_dir / "rgb" / f"{frame:06d}.png"
        if not rgb_path.exists():
            continue

        unity_rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        if "view_bank_az_deg" in row and np.isfinite(float(row.get("view_bank_az_deg", float("nan")))):
            view_plane = float(row["view_bank_az_deg"])
            view_bank_az = view_plane
            view_bank_az_raw = float(row.get("view_bank_az_raw_deg", float("nan")))
        else:
            view_plane_raw = float(row.get("view_plane_deg", float("nan")))
            view_plane = unity_plane_to_bank_az(view_plane_raw)
            view_bank_az = view_plane
            view_bank_az_raw = float("nan")

        distance_m = float(row.get("distance_m", float("nan")))
        if global_scale and np.isfinite(global_distance_m):
            distance_m = global_distance_m

        target_d = estimate_target_diameter_px(
            distance_m,
            fx,
            polyp_diameter_mm,
            scale_boost=scale_boost,
        )

        if bank_blend:
            patch_scaled, az_lo, az_hi, w_hi = build_interpolated_patch(
                view_plane,
                bank_az,
                bank_files,
                target_d,
                cache=rgba_cache,
            )
            bank_az_pick = az_lo if w_hi < 0.5 else az_hi
        else:
            az_lo = az_hi = min(bank_az, key=lambda az: abs(az - view_plane))
            w_hi = 0.0
            bank_az_pick = az_lo
            patch_scaled = resize_patch_rgba(load_rgba(bank_files[bank_az_pick]), target_d)

        anchor_u = float(row.get("anchor_u", cx_default))
        anchor_v = float(row.get("anchor_v", cy_default))
        composite, _paste_mask = paste_rgba_center(
            unity_rgb,
            patch_scaled,
            anchor_u,
            anchor_v,
            lab_match=lab_match,
        )

        stem = f"{frame:06d}"
        comp_path = rgb_out / f"{stem}.png"
        cv2.imwrite(str(comp_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        composite_paths.append(comp_path)

        if write_debug and count < debug_max_frames:
            save_debug_triptych(
                debug_dir / f"match_{frame:06d}.png",
                unity_rgb=unity_rgb,
                patch_rgba=patch_scaled,
                composite_rgb=composite,
                view_plane_deg=view_plane,
                bank_az_lo=az_lo,
                bank_az_hi=az_hi,
                blend_w_hi=w_hi,
                frame=frame,
            )

        manifest_rows.append(
            {
                "frame": frame,
                "view_bank_az_deg": view_bank_az,
                "view_bank_az_raw_deg": view_bank_az_raw,
                "view_plane_deg": view_plane,
                "view_plane_deg_smoothed": smooth_angles,
                "bank_az_lo_deg": az_lo,
                "bank_az_hi_deg": az_hi,
                "bank_blend_weight_hi": w_hi,
                "bank_azimuth_deg": bank_az_pick,
                "anchor_u": anchor_u,
                "anchor_v": anchor_v,
                "distance_m": distance_m,
                "distance_m_global": global_distance_m,
                "target_diameter_px": target_d,
                "composite": str(comp_path.relative_to(out_dir)),
            }
        )
        count += 1

    manifest_csv = out_dir / "composite_manifest.csv"
    if manifest_rows:
        with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    mp4_path = None
    if write_mp4 and composite_paths:
        mp4_path = export_trajectory_mp4(
            out_dir,
            pattern="rgb/*.png",
            fps=mp4_fps,
            output_name="trajectory_composite.mp4",
        )

    summary = {
        "composite_count": count,
        "out_dir": str(out_dir),
        "rgb_dir": str(rgb_out),
        "view_bank": str(view_bank_dir),
        "scale_boost": scale_boost,
        "global_scale": global_scale,
        "global_distance_m": global_distance_m,
        "smooth_window": smooth_window,
        "smooth_angles": smooth_angles,
        "smooth_distance": smooth_distance,
        "bank_blend": bank_blend,
        "write_debug": write_debug,
        "trajectory_mp4": str(mp4_path) if mp4_path else None,
    }
    (out_dir / "composite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_strip(out_dir: Path, *, thumb_h: int = 240, max_cols: int = 16) -> Path:
    rgb_dir = out_dir / "rgb"
    search = rgb_dir if rgb_dir.exists() else out_dir
    comps = sorted(search.glob("*.png"))[:max_cols]
    if not comps:
        comps = sorted(out_dir.glob("frame_*_composite.png"))[:max_cols]
    if not comps:
        raise FileNotFoundError(f"No composites in {out_dir}")
    panels = []
    for path in comps:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = thumb_h / max(h, 1)
        thumb = cv2.resize(img, (max(1, int(w * scale)), thumb_h), interpolation=cv2.INTER_AREA)
        panels.append(thumb)
    strip = panels[0]
    for p in panels[1:]:
        gap = np.full((thumb_h, 8, 3), 12, dtype=np.uint8)
        strip = np.hstack([strip, gap, p])
    out_path = out_dir / "composite_strip.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-consistent gaze polyp composite.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "medical_gan_dataset")
    parser.add_argument("--view-bank", type=Path, default=ROOT / "outputs" / "view_bank")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gaze_composite")
    parser.add_argument("--scale-boost", type=float, default=8.0, help="Visible size multiplier")
    parser.add_argument("--no-global-scale", action="store_true", help="Per-frame distance scaling")
    parser.add_argument("--smooth-window", type=int, default=7, help="Temporal median window (frames)")
    parser.add_argument("--no-smooth-angles", action="store_true")
    parser.add_argument("--no-smooth-distance", action="store_true")
    parser.add_argument("--smooth-anchor", action="store_true")
    parser.add_argument("--no-bank-blend", action="store_true", help="Nearest bank view only")
    parser.add_argument("--no-lab-match", action="store_true")
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Save unity|mesh|composite debug PNGs")
    parser.add_argument("--debug-max-frames", type=int, default=12)
    parser.add_argument("--mp4-fps", type=float, default=15.0)
    args = parser.parse_args()

    summary = composite_dataset(
        args.dataset,
        args.view_bank,
        args.out.resolve(),
        gazing_only=not args.all_frames,
        scale_boost=args.scale_boost,
        global_scale=not args.no_global_scale,
        smooth_window=args.smooth_window,
        smooth_angles=not args.no_smooth_angles,
        smooth_distance=not args.no_smooth_distance,
        smooth_anchor=args.smooth_anchor,
        bank_blend=not args.no_bank_blend,
        lab_match=not args.no_lab_match,
        max_frames=args.max_frames,
        write_mp4=not args.no_mp4,
        write_debug=args.debug,
        debug_max_frames=args.debug_max_frames,
        mp4_fps=args.mp4_fps,
    )
    strip = build_strip(args.out.resolve())
    print("Composites:", summary["composite_count"])
    print("Output:", args.out)
    print("MP4:", summary.get("trajectory_mp4"))
    print("Strip:", strip)


if __name__ == "__main__":
    main()
