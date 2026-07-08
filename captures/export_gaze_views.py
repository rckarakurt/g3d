"""Export gaze anchor and per-frame view angles for gaze-focused Unity captures.

Usage:
  python export_gaze_views.py --dataset captures/medical_gan_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from anchor_from_rays import (
    find_anchor_robust,
    load_positions_and_forwards,
    load_pose_rows as load_ray_pose_rows,
    per_frame_orbit_angle,
    resolve_forward_local,
    theta_to_half_circle_deg,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_GAZE_THRESHOLD_DEG = 45.0
DEFAULT_ANCHOR_INLIER_RADIUS_M = 0.12
DEFAULT_ANCHOR_PATCH_RADIUS_PX = 20
DEFAULT_ANCHOR_PREFILTER_DEG = 35.0


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a / na, b / nb)), -1.0, 1.0))))


def get_look_vector(row: dict) -> np.ndarray:
    if "look_x" in row and row["look_x"]:
        return np.array(
            [float(row["look_x"]), float(row["look_y"]), float(row["look_z"])],
            dtype=np.float64,
        )
    return quat_to_look(
        float(row["rX"]), float(row["rY"]), float(row["rZ"]), float(row["rW"])
    )


def estimate_anchor_robust(
    points: np.ndarray,
    *,
    inlier_radius_m: float = DEFAULT_ANCHOR_INLIER_RADIUS_M,
    iterations: int = 8,
    min_inliers: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust 3D anchor via iterative median on spatial inliers."""
    if len(points) == 0:
        raise ValueError("No points for anchor estimation")
    anchor = np.median(points, axis=0)
    inlier_mask = np.ones(len(points), dtype=bool)
    for _ in range(iterations):
        dists = np.linalg.norm(points - anchor, axis=1)
        inlier_mask = dists <= inlier_radius_m
        if int(inlier_mask.sum()) < min_inliers:
            break
        anchor = np.median(points[inlier_mask], axis=0)
    return anchor, inlier_mask


def load_camera(dataset_dir: Path) -> tuple[np.ndarray, float, int, int]:
    cam = json.loads((dataset_dir / "camera.json").read_text(encoding="utf-8"))
    k = np.array(
        [
            [float(cam["fx"]), 0.0, float(cam["cx"])],
            [0.0, float(cam["fy"]), float(cam["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return k, float(cam["depth_scale"]), int(cam["image_width"]), int(cam["image_height"])


def quat_to_look(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    look = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = np.linalg.norm(look)
    return look / norm if norm > 1e-8 else look


def pose_to_c2w(row: dict) -> np.ndarray:
    rotation = Rotation.from_quat(
        [float(row["rX"]), float(row["rY"]), float(row["rZ"]), float(row["rW"])]
    ).as_matrix()
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation @ np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    c2w[:3, 3] = [float(row["tX"]), float(row["tY"]), float(row["tZ"])]
    return c2w


def unproject_pixel(
    u: float, v: float, depth_m: float, k: np.ndarray, c2w: np.ndarray
) -> np.ndarray:
    cx, cy = k[0, 2], k[1, 2]
    fx, fy = k[0, 0], k[1, 1]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    cam = np.array([x, y, depth_m, 1.0], dtype=np.float64)
    return (c2w @ cam)[:3]


def unproject_center(depth_m: float, k: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    cx, cy = k[0, 2], k[1, 2]
    return unproject_pixel(cx, cy, depth_m, k, c2w)


def unproject_patch_median(
    depth_m: np.ndarray,
    k: np.ndarray,
    c2w: np.ndarray,
    cx: float,
    cy: float,
    *,
    patch_radius_px: int,
    min_valid_pixels: int = 12,
) -> np.ndarray | None:
    """Robust 3D anchor hit from a disk around the image center (polyp-sized region)."""
    h, w = depth_m.shape
    radius = max(1, int(patch_radius_px))
    ix = int(round(cx))
    iy = int(round(cy))
    u0 = max(0, ix - radius)
    u1 = min(w, ix + radius + 1)
    v0 = max(0, iy - radius)
    v1 = min(h, iy + radius + 1)

    patch = depth_m[v0:v1, u0:u1]
    if patch.size == 0:
        return None

    vv, uu = np.mgrid[v0:v1, u0:u1]
    mask = (uu - cx) ** 2 + (vv - cy) ** 2 <= float(radius * radius)
    valid = mask & (patch > 1e-4)
    if int(valid.sum()) < min_valid_pixels:
        return None

    depths = patch[valid]
    # Trim depth outliers inside patch (mucosa vs background speckle).
    lo, hi = np.percentile(depths, [20, 80])
    in_band = valid & (depth_m[v0:v1, u0:u1] >= lo) & (depth_m[v0:v1, u0:u1] <= hi)
    if int(in_band.sum()) < min_valid_pixels:
        in_band = valid

    points = [
        unproject_pixel(float(u), float(v), float(d), k, c2w)
        for v, u, d in zip(vv[in_band], uu[in_band], depth_m[v0:v1, u0:u1][in_band])
    ]
    return np.median(np.stack(points, axis=0), axis=0)


def estimate_normal_from_depth(
    depth_m: np.ndarray,
    k: np.ndarray,
    c2w: np.ndarray,
    cx: float,
    cy: float,
    *,
    patch_radius_px: int = 16,
    step: int = 4,
) -> np.ndarray:
    h, w = depth_m.shape
    ix = int(round(cx))
    iy = int(round(cy))
    radius = max(step, int(patch_radius_px))
    points: list[np.ndarray] = []

    for v in range(max(0, iy - radius), min(h, iy + radius + 1), step):
        for u in range(max(0, ix - radius), min(w, ix + radius + 1), step):
            if (u - cx) ** 2 + (v - cy) ** 2 > float(radius * radius):
                continue
            d = float(depth_m[v, u])
            if d <= 1e-4:
                continue
            points.append(unproject_pixel(float(u), float(v), d, k, c2w))

    center_focus = unproject_patch_median(depth_m, k, c2w, cx, cy, patch_radius_px=min(radius, 24))
    if center_focus is not None:
        points.insert(0, center_focus)

    if len(points) < 3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    p0 = points[0]
    for i in range(1, len(points) - 1):
        v1 = points[i] - p0
        v2 = points[i + 1] - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm > 1e-8:
            n = n / norm
            if np.dot(n, p0) >= 0:
                return n
            return -n
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def build_tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normal / max(np.linalg.norm(normal), 1e-8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(up, n))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent = np.cross(up, n)
    tangent /= max(np.linalg.norm(tangent), 1e-8)
    bitangent = np.cross(n, tangent)
    bitangent /= max(np.linalg.norm(bitangent), 1e-8)
    return tangent, bitangent, n


def angles_in_anchor_frame(
    vector: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray,
) -> tuple[float, float]:
    v = vector / max(np.linalg.norm(vector), 1e-8)
    vt = float(np.dot(v, tangent))
    vb = float(np.dot(v, bitangent))
    vn = float(np.dot(v, normal))
    azimuth = float(np.degrees(np.arctan2(vb, vt)))
    elevation = float(np.degrees(np.arctan2(vn, np.hypot(vt, vb))))
    return azimuth, elevation


def plane_azimuth_0_360(azimuth_deg: float) -> float:
    if not np.isfinite(azimuth_deg):
        return float("nan")
    return float((float(azimuth_deg) + 360.0) % 360.0)


def plane_angle_0_180(azimuth_deg: float) -> float:
    """In-plane viewing angle on anchor tangent plane, range [0, 180]."""
    if not np.isfinite(azimuth_deg):
        return float("nan")
    az = plane_azimuth_0_360(azimuth_deg)
    return az if az <= 180.0 else 360.0 - az


def project_world_point(point: np.ndarray, c2w: np.ndarray, k: np.ndarray) -> tuple[float, float, float]:
    w2c = np.linalg.inv(c2w)
    hom = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    cam = w2c @ hom
    z = float(cam[2])
    if z <= 1e-6:
        return float("nan"), float("nan"), z
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    u = fx * float(cam[0]) / z + cx
    v = fy * float(cam[1]) / z + cy
    return u, v, z


def load_pose_rows(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "poses" / "poses_with_directions.csv"
    if not path.exists():
        path = dataset_dir / "poses" / "poses_per_frame.csv"
    if not path.exists():
        raise FileNotFoundError(f"No pose CSV in {dataset_dir / 'poses'}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            rows.append(dict(item))
    return rows


def read_depth_m(dataset_dir: Path, row: dict, depth_scale: float) -> np.ndarray | None:
    frame = int(float(row["frame"]))
    depth_path = dataset_dir / "depth" / f"{frame:06d}.png"
    if not depth_path.exists():
        return None
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth.astype(np.float32) / depth_scale


def export_gaze_views(
    dataset_dir: Path,
    *,
    gaze_threshold_deg: float = DEFAULT_GAZE_THRESHOLD_DEG,
    frame_start: int = 0,
    anchor_inlier_radius_m: float = DEFAULT_ANCHOR_INLIER_RADIUS_M,
    anchor_patch_radius_px: int = DEFAULT_ANCHOR_PATCH_RADIUS_PX,
    anchor_prefilter_deg: float = DEFAULT_ANCHOR_PREFILTER_DEG,
    write_plot: bool = False,
) -> dict:
    dataset_dir = dataset_dir.resolve()
    k, depth_scale, width, height = load_camera(dataset_dir)
    cx, cy = k[0, 2], k[1, 2]
    pose_rows = load_pose_rows(dataset_dir)

    focus_points: list[np.ndarray] = []
    per_frame_focus: list[np.ndarray | None] = []
    calibration_records: list[dict] = []

    for row in pose_rows:
        frame = int(float(row["frame"]))
        depth_m = read_depth_m(dataset_dir, row, depth_scale)
        c2w = pose_to_c2w(row)
        if depth_m is None:
            per_frame_focus.append(None)
            continue
        focus = unproject_patch_median(
            depth_m, k, c2w, cx, cy, patch_radius_px=anchor_patch_radius_px
        )
        if focus is None:
            per_frame_focus.append(None)
            continue
        per_frame_focus.append(focus)
        if frame >= frame_start:
            look = get_look_vector(row)
            cam_pos = c2w[:3, 3]
            focus_points.append(focus)
            calibration_records.append(
                {
                    "focus": focus,
                    "look": look,
                    "cam_pos": cam_pos,
                    "normal": estimate_normal_from_depth(
                        depth_m,
                        k,
                        c2w,
                        cx,
                        cy,
                        patch_radius_px=anchor_patch_radius_px,
                    ),
                }
            )

    if not focus_points:
        raise RuntimeError(
            f"No valid depth patch hits from frame {frame_start} onward. "
            "Check depth at image center."
        )

    ray_rows = load_ray_pose_rows(dataset_dir)
    positions, forwards, ray_frame_ids = load_positions_and_forwards(
        ray_rows, frame_start=frame_start
    )
    _, forwards, _ = resolve_forward_local(positions, forwards)
    anchor_pos, _ray_inlier_mask, ray_dev_all = find_anchor_robust(
        positions,
        forwards,
        max_dev_deg=anchor_prefilter_deg,
    )
    orbit_theta, plane_normal, u_axis, v_axis = per_frame_orbit_angle(positions, anchor_pos)
    orbit_theta_180 = theta_to_half_circle_deg(orbit_theta)
    ray_frame_map = {fid: i for i, fid in enumerate(ray_frame_ids)}

    anchor_normal = plane_normal / max(np.linalg.norm(plane_normal), 1e-8)
    tangent, bitangent, normal = u_axis, v_axis, anchor_normal

    poses_dir = dataset_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    gaze_rows: list[dict] = []
    gazing_count = 0

    for row, focus in zip(pose_rows, per_frame_focus):
        frame = int(float(row["frame"]))
        image_name = row.get("image_name", f"image_{frame:04d}.png")
        c2w = pose_to_c2w(row)
        look = get_look_vector(row)
        cam_pos = c2w[:3, 3]

        to_global = anchor_pos - cam_pos
        distance_m = float(np.linalg.norm(to_global))
        to_global_dir = (
            to_global / distance_m if distance_m > 1e-6 else np.zeros(3, dtype=np.float64)
        )

        view_azimuth_deg = float("nan")
        view_elevation_deg = float("nan")
        view_plane_az_deg = float("nan")
        view_plane_deg = float("nan")
        view_tilt_deg = float("nan")
        ray_dev_deg = float("nan")
        if frame in ray_frame_map:
            ri = ray_frame_map[frame]
            view_azimuth_deg = float(orbit_theta[ri])
            view_plane_deg = float(orbit_theta_180[ri])
            view_plane_az_deg = plane_azimuth_0_360(view_azimuth_deg)
            ray_dev_deg = float(ray_dev_all[ri])
            if distance_m > 1e-6:
                view_elevation_deg, _ = angles_in_anchor_frame(
                    to_global_dir, tangent, bitangent, normal
                )
                view_tilt_deg = view_elevation_deg

        # Image anchor = screen center (user holds target on crosshair).
        anchor_u, anchor_v = float(cx), float(cy)
        proj_focus_u, proj_focus_v, _ = (
            project_world_point(focus, c2w, k) if focus is not None else (float("nan"), float("nan"), float("nan"))
        )
        global_anchor_u, global_anchor_v, _ = project_world_point(anchor_pos, c2w, k)
        reproj_error_px = float(np.hypot(proj_focus_u - cx, proj_focus_v - cy))
        global_reproj_error_px = float(np.hypot(global_anchor_u - cx, global_anchor_v - cy))

        gaze_error_deg = angle_between_deg(look, to_global)
        focus_drift_m = float("nan")
        center_ray_error_deg = float("nan")
        if focus is not None:
            focus_drift_m = float(np.linalg.norm(focus - anchor_pos))
            center_ray_error_deg = angle_between_deg(look, focus - cam_pos)

        has_focus = focus is not None
        in_image = True
        is_gazing = (
            frame >= frame_start
            and frame in ray_frame_map
            and np.isfinite(ray_dev_deg)
            and ray_dev_deg <= gaze_threshold_deg
            and has_focus
            and in_image
        )
        if is_gazing:
            gazing_count += 1

        row_out = {
            "frame": frame,
            "image_name": image_name,
            "anchor_u": anchor_u,
            "anchor_v": anchor_v,
            "proj_focus_u": proj_focus_u,
            "proj_focus_v": proj_focus_v,
            "global_anchor_u": global_anchor_u,
            "global_anchor_v": global_anchor_v,
            "reproj_error_px": reproj_error_px,
            "global_reproj_error_px": global_reproj_error_px,
            "distance_m": distance_m,
            "view_azimuth_deg": view_azimuth_deg,
            "view_elevation_deg": view_elevation_deg,
            "view_plane_az_deg": view_plane_az_deg,
            "view_plane_deg": view_plane_deg,
            "view_tilt_deg": view_tilt_deg,
            "gaze_error_deg": gaze_error_deg,
            "ray_dev_deg": ray_dev_deg,
            "center_ray_error_deg": center_ray_error_deg,
            "focus_drift_m": focus_drift_m,
            "is_gazing": int(is_gazing),
            "tX": float(row["tX"]),
            "tY": float(row["tY"]),
            "tZ": float(row["tZ"]),
            "look_x": float(look[0]),
            "look_y": float(look[1]),
            "look_z": float(look[2]),
        }
        if has_focus:
            row_out["focus_wx"] = float(focus[0])
            row_out["focus_wy"] = float(focus[1])
            row_out["focus_wz"] = float(focus[2])
        else:
            row_out["focus_wx"] = float("nan")
            row_out["focus_wy"] = float("nan")
            row_out["focus_wz"] = float("nan")
        gaze_rows.append(row_out)

    inlier_focus = [
        rec["focus"]
        for rec in calibration_records
        if float(np.linalg.norm(rec["focus"] - anchor_pos)) <= anchor_inlier_radius_m
    ]
    ray_dev_mean = float(np.mean(ray_dev_all))
    focus_std_m = float(
        np.linalg.norm(np.std(np.stack(inlier_focus or focus_points, axis=0), axis=0))
    )
    anchor_meta = {
        "anchor_pos": anchor_pos.tolist(),
        "anchor_normal": anchor_normal.tolist(),
        "tangent": tangent.tolist(),
        "bitangent": bitangent.tolist(),
        "anchor_mode": "screen_center_image + ray_3d_angles",
        "anchor_note": (
            "anchor_u/v = image center (crosshair) — composite placement. "
            "anchor_pos 3D = Unity ray intersection (view_plane_deg only). "
            "reproj_error_px = depth patch vs center; global_reproj = ray 3D vs center."
        ),
        "gaze_threshold_deg": gaze_threshold_deg,
        "frame_start": frame_start,
        "anchor_inlier_radius_m": anchor_inlier_radius_m,
        "anchor_spatial_radius_m": anchor_inlier_radius_m,
        "anchor_patch_radius_px": anchor_patch_radius_px,
        "anchor_prefilter_deg": anchor_prefilter_deg,
        "ray_dev_deg_mean": ray_dev_mean,
        "ray_dev_deg_max": float(np.max(ray_dev_all)),
        "orbit_theta_span_deg": [
            float(np.min(orbit_theta_180)),
            float(np.max(orbit_theta_180)),
        ],
        "anchor_calibration_frames": len(focus_points),
        "anchor_inlier_frames": len(inlier_focus),
        "frame_count": len(pose_rows),
        "gazing_frame_count": gazing_count,
        "focus_std_m": focus_std_m,
        "image_size": [width, height],
        "principal_point": [float(cx), float(cy)],
    }
    (poses_dir / "focus_anchor.json").write_text(
        json.dumps(anchor_meta, indent=2), encoding="utf-8"
    )

    csv_path = poses_dir / "gaze_views.csv"
    fieldnames = list(gaze_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gaze_rows)

    if write_plot:
        plot_gaze_analysis(dataset_dir, gaze_rows, anchor_meta)
    return anchor_meta


def plot_gaze_analysis(dataset_dir: Path, gaze_rows: list[dict], anchor_meta: dict) -> None:
    traj_dir = dataset_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)

    gazing = [r for r in gaze_rows if int(r["is_gazing"]) == 1]
    if not gazing:
        gazing = gaze_rows

    az = [float(r["view_azimuth_deg"]) for r in gazing if np.isfinite(float(r["view_azimuth_deg"]))]
    el = [float(r["view_elevation_deg"]) for r in gazing if np.isfinite(float(r["view_elevation_deg"]))]
    err = [float(r["gaze_error_deg"]) for r in gaze_rows if np.isfinite(float(r["gaze_error_deg"]))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    if az and el:
        axes[0].scatter(az, el, c="#2563eb", s=18, alpha=0.75)
        axes[0].set_xlabel("Azimuth (deg)")
        axes[0].set_ylabel("Elevation (deg)")
        axes[0].set_title(f"Gaze angles @ anchor (n={len(az)})")
        axes[0].grid(True, alpha=0.3)

    axes[1].plot(err, color="#16a34a", linewidth=1.0)
    axes[1].axhline(float(anchor_meta["gaze_threshold_deg"]), color="#dc2626", ls="--", lw=1)
    frame_start = int(anchor_meta.get("frame_start", 0))
    if frame_start > 0:
        axes[1].axvline(frame_start, color="#9333ea", ls=":", lw=1.2, label=f"frame_start={frame_start}")
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Gaze error (deg)")
    axes[1].set_title("Look vs anchor direction")
    axes[1].grid(True, alpha=0.3)

    au = [float(r["anchor_u"]) for r in gazing if np.isfinite(float(r["anchor_u"]))]
    av = [float(r["anchor_v"]) for r in gazing if np.isfinite(float(r["anchor_v"]))]
    if au and av:
        axes[2].scatter(au, av, c="#16a34a", s=18, alpha=0.75, label="projected anchor")
        pcx, pcy = anchor_meta["principal_point"]
        axes[2].scatter([pcx], [pcy], c="#eab308", s=40, marker="x", label="principal point")
        axes[2].set_xlim(0, anchor_meta["image_size"][0])
        axes[2].set_ylim(anchor_meta["image_size"][1], 0)
        axes[2].set_xlabel("u (px)")
        axes[2].set_ylabel("v (px)")
        axes[2].set_title("Computed anchor projection (green)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(traj_dir / "gaze_analysis.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gaze anchor and view angles.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "medical_gan_dataset",
        help="medical_gan_dataset folder",
    )
    parser.add_argument(
        "--gaze-threshold-deg",
        type=float,
        default=DEFAULT_GAZE_THRESHOLD_DEG,
        help="Max angle between look and anchor to count as gazing",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="First frame used for anchor calibration and is_gazing (e.g. 53 if focus begins there)",
    )
    parser.add_argument(
        "--anchor-inlier-radius-m",
        type=float,
        default=DEFAULT_ANCHOR_INLIER_RADIUS_M,
        help="3D spatial radius of anchor region / polyp footprint (meters)",
    )
    parser.add_argument(
        "--anchor-patch-radius-px",
        type=int,
        default=DEFAULT_ANCHOR_PATCH_RADIUS_PX,
        help="Image disk radius for depth patch unproject (pixels)",
    )
    parser.add_argument(
        "--anchor-prefilter-deg",
        type=float,
        default=DEFAULT_ANCHOR_PREFILTER_DEG,
        help="Max look vs center-ray angle for anchor calibration frames",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write trajectory/gaze_analysis.png (off by default)",
    )
    args = parser.parse_args()

    meta = export_gaze_views(
        args.dataset,
        gaze_threshold_deg=args.gaze_threshold_deg,
        frame_start=args.frame_start,
        anchor_inlier_radius_m=args.anchor_inlier_radius_m,
        anchor_patch_radius_px=args.anchor_patch_radius_px,
        anchor_prefilter_deg=args.anchor_prefilter_deg,
        write_plot=args.plot,
    )
    print(f"Gaze export: {args.dataset / 'poses' / 'gaze_views.csv'}")
    print(f"  frames: {meta['frame_count']}, anchor from frame: {meta['frame_start']}")
    print(f"  anchor calibration frames: {meta['anchor_calibration_frames']}")
    print(f"  anchor prefilter/inlier frames: {meta['anchor_inlier_frames']}/{meta['anchor_calibration_frames']}")
    print(f"  ray_dev_deg_mean: {meta.get('ray_dev_deg_mean', float('nan')):.3f}")
    print(f"  orbit span (0-180): {meta.get('orbit_theta_span_deg')}")
    print(f"  gazing: {meta['gazing_frame_count']}")
    print(f"  focus_std_m: {meta['focus_std_m']:.5f}")
    print(f"  anchor: {meta['anchor_pos']}")


if __name__ == "__main__":
    main()
