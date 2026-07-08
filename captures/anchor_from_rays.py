"""Anchor point from camera ray intersection (Unity world, no depth required).

The anchor is the 3D point closest to all view rays (position + forward).
Orbit angles come from PCA of camera positions around that anchor.

Usage:
  python anchor_from_rays.py --dataset captures/medical_gan_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_FORWARD_LOCAL = (0.0, 0.0, 1.0)


def quat_xyzw_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unity quaternion (xyzw) -> 3x3 rotation (camera local -> world)."""
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def load_pose_rows(dataset_dir: Path) -> list[dict]:
    for name in ("gan_training_index.csv", "poses/poses_per_frame.csv", "poses/poses_with_directions.csv"):
        path = dataset_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError(f"No pose CSV under {dataset_dir}")


def load_positions_and_forwards(
    rows: list[dict],
    *,
    forward_local: tuple[float, float, float] = DEFAULT_FORWARD_LOCAL,
    frame_start: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    positions: list[np.ndarray] = []
    forwards: list[np.ndarray] = []
    frame_ids: list[int] = []

    f_local = np.array(forward_local, dtype=np.float64)
    for row in rows:
        if str(row.get("has_pose", "1")) not in ("1", "True", "true", ""):
            continue
        frame = int(float(row["frame"]))
        if frame < frame_start:
            continue
        p = np.array([float(row["tX"]), float(row["tY"]), float(row["tZ"])], dtype=np.float64)
        R = quat_xyzw_to_R(
            float(row.get("rX", 0.0)),
            float(row.get("rY", 0.0)),
            float(row.get("rZ", 0.0)),
            float(row.get("rW", 1.0)),
        )
        fwd = R @ f_local
        fwd /= max(np.linalg.norm(fwd), 1e-12)
        positions.append(p)
        forwards.append(fwd)
        frame_ids.append(frame)

    if not positions:
        raise RuntimeError("No pose rows after frame_start filter.")
    return np.stack(positions, axis=0), np.stack(forwards, axis=0), frame_ids


def find_anchor_point(positions: np.ndarray, forwards: np.ndarray) -> np.ndarray:
    """Least-squares closest point to a set of view rays."""
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for origin, direction in zip(positions, forwards):
        proj = identity - np.outer(direction, direction)
        A += proj
        b += proj @ origin
    anchor, *_ = np.linalg.lstsq(A, b, rcond=None)
    return anchor


def anchor_sanity_check(
    positions: np.ndarray,
    forwards: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    """Per-frame angle (deg) between forward and direction to anchor."""
    to_anchor = anchor[None, :] - positions
    to_anchor_n = to_anchor / (np.linalg.norm(to_anchor, axis=1, keepdims=True) + 1e-12)
    cos_dev = np.clip(np.sum(to_anchor_n * forwards, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos_dev))


def find_anchor_robust(
    positions: np.ndarray,
    forwards: np.ndarray,
    *,
    max_dev_deg: float = 20.0,
    iterations: int = 4,
    min_rays: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively drop outlier rays then re-fit anchor."""
    mask = np.ones(len(positions), dtype=bool)
    anchor = find_anchor_point(positions, forwards)
    dev_deg = anchor_sanity_check(positions, forwards, anchor)

    for _ in range(iterations):
        dev_deg = anchor_sanity_check(positions, forwards, anchor)
        new_mask = dev_deg <= max_dev_deg
        if int(new_mask.sum()) < min_rays:
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        anchor = find_anchor_point(positions[mask], forwards[mask])

    dev_deg = anchor_sanity_check(positions, forwards, anchor)
    return anchor, mask, dev_deg


def per_frame_orbit_angle(
    positions: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA orbit plane; angle of each camera around anchor in that plane (degrees)."""
    centered = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u_axis, v_axis, plane_normal = vt[0], vt[1], vt[2]
    rel = positions - anchor
    theta = np.degrees(np.arctan2(rel @ v_axis, rel @ u_axis))
    return theta, plane_normal, u_axis, v_axis


def theta_to_half_circle_deg(theta: np.ndarray) -> np.ndarray:
    """Map orbit theta to continuous [0, 180] span for semicircle orbit."""
    t = np.unwrap(np.radians(theta))
    t = np.degrees(t)
    t -= float(np.min(t))
    if float(np.max(t)) > 180.0:
        t = t * (180.0 / float(np.max(t)))
    return t


def resolve_forward_local(
    positions: np.ndarray,
    forwards: np.ndarray,
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    """Pick +Z or -Z local forward whichever gives lower mean deviation."""
    best_fwd = DEFAULT_FORWARD_LOCAL
    best_dev = anchor_sanity_check(positions, forwards, find_anchor_point(positions, forwards))
    mean_dev = float(np.mean(best_dev))

    if mean_dev > 90.0:
        alt_rows_fwd = -forwards
        anchor_alt = find_anchor_point(positions, alt_rows_fwd)
        dev_alt = anchor_sanity_check(positions, alt_rows_fwd, anchor_alt)
        if float(np.mean(dev_alt)) < mean_dev:
            return (0.0, 0.0, -1.0), alt_rows_fwd, dev_alt
    return best_fwd, forwards, best_dev


def export_anchor_from_rays(
    dataset_dir: Path,
    *,
    frame_start: int = 0,
    max_dev_deg: float = 20.0,
    output_npz: Path | None = None,
) -> dict:
    dataset_dir = dataset_dir.resolve()
    rows = load_pose_rows(dataset_dir)
    positions, forwards, frame_ids = load_positions_and_forwards(rows, frame_start=frame_start)

    forward_local, forwards, _ = resolve_forward_local(positions, forwards)
    anchor, inlier_mask, dev_deg = find_anchor_robust(
        positions, forwards, max_dev_deg=max_dev_deg
    )
    theta, plane_normal, u_axis, v_axis = per_frame_orbit_angle(positions, anchor)
    theta_180 = theta_to_half_circle_deg(theta)

    meta = {
        "anchor_pos": anchor.tolist(),
        "anchor_mode": "ray_intersection_unity",
        "forward_local": list(forward_local),
        "frame_start": frame_start,
        "ray_count": len(frame_ids),
        "ray_inlier_count": int(inlier_mask.sum()),
        "dev_deg_mean": float(np.mean(dev_deg)),
        "dev_deg_median": float(np.median(dev_deg)),
        "dev_deg_max": float(np.max(dev_deg)),
        "theta_min_deg": float(np.min(theta)),
        "theta_max_deg": float(np.max(theta)),
        "theta_180_min_deg": float(np.min(theta_180)),
        "theta_180_max_deg": float(np.max(theta_180)),
        "plane_normal": plane_normal.tolist(),
        "u_axis": u_axis.tolist(),
        "v_axis": v_axis.tolist(),
    }

    poses_dir = dataset_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame CSV aligned to all dataset frames
    frame_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    all_frames = sorted({int(float(r["frame"])) for r in rows})
    csv_path = poses_dir / "gaze_views_rays.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "ray_dev_deg",
                "ray_inlier",
                "orbit_theta_deg",
                "view_plane_deg",
                "tX",
                "tY",
                "tZ",
            ],
        )
        writer.writeheader()
        for row in rows:
            frame = int(float(row["frame"]))
            if frame not in frame_to_idx:
                continue
            i = frame_to_idx[frame]
            writer.writerow(
                {
                    "frame": frame,
                    "ray_dev_deg": float(dev_deg[i]),
                    "ray_inlier": int(inlier_mask[i]),
                    "orbit_theta_deg": float(theta[i]),
                    "view_plane_deg": float(theta_180[i]),
                    "tX": float(row["tX"]),
                    "tY": float(row["tY"]),
                    "tZ": float(row["tZ"]),
                }
            )

    anchor_json = poses_dir / "focus_anchor_rays.json"
    anchor_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    npz_path = output_npz or (poses_dir / "anchor_point_rays.npz")
    np.savez(
        npz_path,
        anchor=anchor,
        positions=positions,
        forwards=forwards,
        theta=theta,
        theta_180=theta_180,
        dev_deg=dev_deg,
        inlier_mask=inlier_mask,
        frame_ids=np.array(frame_ids, dtype=np.int32),
        plane_normal=plane_normal,
        u_axis=u_axis,
        v_axis=v_axis,
    )
    meta["outputs"] = {
        "csv": str(csv_path.relative_to(dataset_dir)),
        "json": str(anchor_json.relative_to(dataset_dir)),
        "npz": str(npz_path.relative_to(dataset_dir)),
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor from camera ray intersection (Unity pose only).")
    parser.add_argument("--dataset", type=Path, default=ROOT / "medical_gan_dataset")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--max-dev-deg", type=float, default=20.0)
    args = parser.parse_args()

    meta = export_anchor_from_rays(
        args.dataset,
        frame_start=args.frame_start,
        max_dev_deg=args.max_dev_deg,
    )
    print(f"Anchor (Unity world): {meta['anchor_pos']}")
    print(
        f"Ray deviation deg: mean={meta['dev_deg_mean']:.2f}  "
        f"median={meta['dev_deg_median']:.2f}  max={meta['dev_deg_max']:.2f}"
    )
    print(f"Inlier rays: {meta['ray_inlier_count']} / {meta['ray_count']}")
    print(
        f"Orbit theta: {meta['theta_min_deg']:.1f} .. {meta['theta_max_deg']:.1f} deg  "
        f"(plane 0-180: {meta['theta_180_min_deg']:.1f} .. {meta['theta_180_max_deg']:.1f})"
    )
    print(f"Saved: {meta['outputs']}")


if __name__ == "__main__":
    main()
