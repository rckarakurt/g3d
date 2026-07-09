"""Trajectory map colored by gaze / viewing angles.

Usage:
  python export_gaze_trajectory_map.py --dataset medical_gan_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def load_gaze_rows(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "poses" / "gaze_views.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run export_gaze_views.py first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_pose_rows(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "poses" / "poses_per_frame.csv"
    if not path.exists():
        path = dataset_dir / "poses" / "poses_with_directions.csv"
    if not path.exists():
        raise FileNotFoundError(f"No pose CSV in {dataset_dir / 'poses'}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quat_to_look(row: dict) -> np.ndarray:
    if row.get("look_x"):
        return np.array(
            [float(row["look_x"]), float(row["look_y"]), float(row["look_z"])],
            dtype=np.float64,
        )
    rotation = Rotation.from_quat(
        [float(row["rX"]), float(row["rY"]), float(row["rZ"]), float(row["rW"])]
    ).as_matrix()
    look = rotation @ np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(look)
    return look / n if n > 1e-8 else look


def merge_gaze_poses(gaze_rows: list[dict], pose_rows: list[dict]) -> list[dict]:
    pose_by_frame = {int(float(r["frame"])): r for r in pose_rows}
    merged: list[dict] = []
    for g in gaze_rows:
        frame = int(float(g["frame"]))
        p = pose_by_frame.get(frame)
        if p is None:
            continue
        merged.append({**p, **g})
    return merged


def export_gaze_trajectory_map(
    dataset_dir: Path,
    *,
    out_path: Path | None = None,
    angle_field: str = "view_bank_az_deg",
    dpi: int = 180,
) -> Path:
    dataset_dir = dataset_dir.resolve()
    gaze_rows = load_gaze_rows(dataset_dir)
    pose_rows = load_pose_rows(dataset_dir)
    anchor_meta = json.loads((dataset_dir / "poses" / "focus_anchor.json").read_text(encoding="utf-8"))
    rows = merge_gaze_poses(gaze_rows, pose_rows)
    if not rows:
        raise RuntimeError("No merged gaze+pose rows.")

    positions = np.array([[float(r["tX"]), float(r["tY"]), float(r["tZ"])] for r in rows], dtype=np.float64)
    origin = positions[0].copy()
    rel = positions - origin

    angles = np.array([float(r.get(angle_field, float("nan"))) for r in rows], dtype=np.float64)
    valid_angle = np.isfinite(angles)
    if not valid_angle.any():
        angles = np.array([float(r.get("view_azimuth_deg", 0)) for r in rows], dtype=np.float64)
        angle_field = "view_azimuth_deg"

    anchor = np.array(anchor_meta["anchor_pos"], dtype=np.float64) - origin
    gazing = np.array([int(r.get("is_gazing", 0)) for r in rows], dtype=np.int32)

    cmap = plt.cm.plasma
    norm = Normalize(vmin=float(np.nanmin(angles)), vmax=float(np.nanmax(angles)))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.28)

    # --- XZ path colored by viewing angle ---
    ax_xz = fig.add_subplot(gs[0, 0])
    pts_xz = np.stack([rel[:, 0], rel[:, 2]], axis=1)
    segs = np.stack([pts_xz[:-1], pts_xz[1:]], axis=1)
    seg_colors = cmap(norm(angles[:-1]))
    ax_xz.add_collection(LineCollection(segs, colors=seg_colors, linewidths=2.5, alpha=0.95))
    sc = ax_xz.scatter(
        rel[:, 0], rel[:, 2], c=angles, cmap=cmap, norm=norm, s=22, edgecolors="white", linewidths=0.3, zorder=3
    )
    ax_xz.scatter(anchor[0], anchor[2], c="#f59e0b", s=220, marker="*", edgecolors="#92400e", linewidths=1.2, zorder=5, label="anchor")
    ax_xz.scatter(0, 0, c="#2563eb", s=60, marker="o", zorder=4, label="start")
    ax_xz.set_xlabel("X (m, rel.)")
    ax_xz.set_ylabel("Z (m, rel.)")
    ax_xz.set_title(f"Camera path XZ — color = {angle_field}")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.25)
    ax_xz.legend(loc="upper right", fontsize=8)

    # --- XY path ---
    ax_xy = fig.add_subplot(gs[0, 1])
    pts_xy = np.stack([rel[:, 0], rel[:, 1]], axis=1)
    segs_xy = np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)
    ax_xy.add_collection(LineCollection(segs_xy, colors=seg_colors, linewidths=2.5, alpha=0.95))
    ax_xy.scatter(rel[:, 0], rel[:, 1], c=angles, cmap=cmap, norm=norm, s=22, edgecolors="white", linewidths=0.3)
    ax_xy.scatter(anchor[0], anchor[1], c="#f59e0b", s=220, marker="*", edgecolors="#92400e", linewidths=1.2)
    ax_xy.scatter(0, 0, c="#2563eb", s=60, marker="o")
    ax_xy.set_xlabel("X (m, rel.)")
    ax_xy.set_ylabel("Y (m, rel.)")
    ax_xy.set_title("Camera path XY")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.25)

    # --- YZ path (2D instead of 3D for Windows stability) ---
    ax_yz = fig.add_subplot(gs[0, 2])
    pts_yz = np.stack([rel[:, 1], rel[:, 2]], axis=1)
    segs_yz = np.stack([pts_yz[:-1], pts_yz[1:]], axis=1)
    ax_yz.add_collection(LineCollection(segs_yz, colors=seg_colors, linewidths=2.5, alpha=0.95))
    ax_yz.scatter(rel[:, 1], rel[:, 2], c=angles, cmap=cmap, norm=norm, s=22, edgecolors="white", linewidths=0.3)
    ax_yz.scatter(anchor[1], anchor[2], c="#f59e0b", s=220, marker="*", edgecolors="#92400e", linewidths=1.2)
    ax_yz.scatter(rel[0, 1], rel[0, 2], c="#2563eb", s=60, marker="o")
    ax_yz.set_xlabel("Y (m, rel.)")
    ax_yz.set_ylabel("Z (m, rel.)")
    ax_yz.set_title("Camera path YZ")
    ax_yz.set_aspect("equal", adjustable="box")
    ax_yz.grid(True, alpha=0.25)

    # --- Angle vs frame ---
    ax_t = fig.add_subplot(gs[1, 0])
    frames = [int(float(r["frame"])) for r in rows]
    ax_t.plot(frames, angles, color="#2563eb", linewidth=1.2, label=angle_field)
    if "view_elevation_deg" in rows[0]:
        el = [float(r["view_elevation_deg"]) for r in rows]
        ax_t.plot(frames, el, color="#16a34a", linewidth=0.9, alpha=0.75, label="view_elevation_deg")
    ax_t.scatter(
        [f for f, g in zip(frames, gazing) if g == 0],
        [a for a, g in zip(angles, gazing) if g == 0],
        c="#dc2626",
        s=20,
        label="not gazing",
        zorder=3,
    )
    ax_t.set_xlabel("Frame")
    ax_t.set_ylabel("Angle (deg)")
    ax_t.set_title("Viewing angle over time")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)

    # --- Azimuth vs elevation ---
    ax_ae = fig.add_subplot(gs[1, 1])
    az = [float(r["view_azimuth_deg"]) for r in rows]
    el = [float(r["view_elevation_deg"]) for r in rows]
    ax_ae.scatter(az, el, c=frames, cmap="viridis", s=28, alpha=0.85)
    ax_ae.set_xlabel("view_azimuth_deg")
    ax_ae.set_ylabel("view_elevation_deg")
    ax_ae.set_title("Angle space (color = frame)")
    ax_ae.grid(True, alpha=0.3)

    # --- Look arrows XZ toward anchor ---
    ax_look = fig.add_subplot(gs[1, 2])
    ax_look.plot(rel[:, 0], rel[:, 2], color="#94a3b8", linewidth=0.8, alpha=0.5)
    stride = max(1, len(rows) // 24)
    for i in range(0, len(rows), stride):
        look = quat_to_look(rows[i])
        o = rel[i]
        end = o + look * 0.12
        ax_look.arrow(
            o[0], o[2], end[0] - o[0], end[2] - o[2],
            head_width=0.015, head_length=0.02, fc=cmap(norm(angles[i])), ec=cmap(norm(angles[i])), alpha=0.9,
        )
    ax_look.scatter(anchor[0], anchor[2], c="#f59e0b", s=180, marker="*", edgecolors="#92400e", zorder=5)
    ax_look.set_xlabel("X (m, rel.)")
    ax_look.set_ylabel("Z (m, rel.)")
    ax_look.set_title("Look direction (XZ) → anchor")
    ax_look.set_aspect("equal", adjustable="box")
    ax_look.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=[ax_xz, ax_xy], fraction=0.025, pad=0.02)
    cbar.set_label(f"{angle_field} (deg)")

    span = anchor_meta.get("orbit_theta_span_deg", [float(np.nanmin(angles)), float(np.nanmax(angles))])
    fig.suptitle(
        f"Gaze trajectory map — {len(rows)} frames, "
        f"{angle_field} {float(np.nanmin(angles)):.1f}°..{float(np.nanmax(angles)):.1f}° "
        f"(orbit span {span[0]:.1f}°..{span[1]:.1f}°)",
        fontsize=13,
        y=0.98,
    )

    traj_dir = dataset_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path or traj_dir / "gaze_trajectory_map.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "frame_count": len(rows),
        "gazing_count": int(gazing.sum()),
        "angle_field": angle_field,
        "angle_min_deg": float(np.nanmin(angles)),
        "angle_max_deg": float(np.nanmax(angles)),
        "orbit_theta_span_deg": span,
        "output_png": str(out_path),
    }
    (traj_dir / "gaze_trajectory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gaze-colored trajectory map.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "medical_gan_dataset")
    parser.add_argument(
        "--angle-field",
        type=str,
        default="view_bank_az_deg",
        choices=(
            "view_bank_az_deg",
            "view_bank_az_raw_deg",
            "view_plane_deg",
            "view_azimuth_deg",
            "view_elevation_deg",
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out = export_gaze_trajectory_map(
        args.dataset,
        out_path=args.output,
        angle_field=args.angle_field,
    )
    print("Saved:", out)


if __name__ == "__main__":
    main()
