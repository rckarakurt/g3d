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


def wrap_angle_deg(angle: float) -> float:
    """Wrap to (-180, 180]."""
    if not np.isfinite(angle):
        return float("nan")
    a = (float(angle) + 180.0) % 360.0 - 180.0
    return float(a)


def fold_bank_azimuth(az_deg: float) -> float:
    """Orbit duzlemi acisini mesh view bank yarim dairesine (-90..+90) katla."""
    if not np.isfinite(az_deg):
        return float("nan")
    az = wrap_angle_deg(float(az_deg))
    if az > 90.0:
        return float(180.0 - az)
    if az < -90.0:
        return float(-180.0 - az)
    return az


def bank_azimuth_y_axis(anchor: np.ndarray, cam_pos: np.ndarray) -> float:
    """View-bank azimuth: Y ekseni (dikey), XZ duzlemi.

    0 = kamera +Z tarafinda (duz yuz), +90 = +X yan, -90 = -X yan.
    Mesh turntable (mesh_rotate_y) ile ayni konvansiyon.
    """
    rel = np.asarray(cam_pos, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    x = float(rel[0])
    z = float(rel[2])
    if np.hypot(x, z) < 1e-9:
        return 0.0
    return wrap_angle_deg(float(np.degrees(np.arctan2(x, z))))


def build_orbit_frame(
    anchor: np.ndarray,
    positions: np.ndarray,
    forwards: np.ndarray,
    *,
    inlier_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orbit duzleminde (front, right, orbit_axis) — mesh Y turntable ile eslesir.

    front: anchor -> frontal kamera (0 deg)
    right: front x orbit_axis
    orbit_axis: kapsul yarim daire ekseni (PCA, Unity Y'ye yakin tutulur)
    """
    mask = np.ones(len(positions), dtype=bool) if inlier_mask is None else inlier_mask
    rel = positions - anchor
    rel_fit = rel[mask] if int(mask.sum()) >= 3 else rel

    _, _, vt = np.linalg.svd(rel_fit, full_matrices=False)
    u_axis, _v_axis, orbit_axis = vt[0], vt[1], vt[2]

    unity_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(orbit_axis, unity_y))) < 0.5:
        # Egik orbit: en kucuk singular vektor yerine en buyuk disi haric tut
        orbit_axis = vt[2]
    if float(np.dot(orbit_axis, unity_y)) < 0.0:
        orbit_axis = -orbit_axis

    ref_cam = pick_frontal_reference_cam(
        positions, forwards, anchor, inlier_mask=mask
    )
    ref_rel = ref_cam - anchor
    ref_rel = ref_rel - np.dot(ref_rel, orbit_axis) * orbit_axis
    ref_norm = float(np.linalg.norm(ref_rel))
    if ref_norm < 1e-9:
        front = u_axis - np.dot(u_axis, orbit_axis) * orbit_axis
        front /= max(np.linalg.norm(front), 1e-12)
    else:
        front = ref_rel / ref_norm

    right = np.cross(orbit_axis, front)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        right = u_axis - np.dot(u_axis, orbit_axis) * orbit_axis
        right /= max(np.linalg.norm(right), 1e-12)
    else:
        right = right / right_norm

    front = np.cross(right, orbit_axis)
    front /= max(np.linalg.norm(front), 1e-12)
    return front, right, orbit_axis


def bank_azimuth_orbit_plane(
    anchor: np.ndarray,
    cam_pos: np.ndarray,
    front: np.ndarray,
    right: np.ndarray,
    orbit_axis: np.ndarray,
) -> float:
    """Anchor -> kamera vektorunun orbit duzlemindeki acisi (0=frontal)."""
    rel = np.asarray(cam_pos, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    rel = rel - np.dot(rel, orbit_axis) * orbit_axis
    x = float(np.dot(rel, right))
    z = float(np.dot(rel, front))
    if np.hypot(x, z) < 1e-9:
        return 0.0
    return wrap_angle_deg(float(np.degrees(np.arctan2(x, z))))


def project_to_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """v' = v - (v·k)k — vektoru k'ye dik duzleme iz dusur."""
    k = np.asarray(plane_normal, dtype=np.float64)
    k = k / max(np.linalg.norm(k), 1e-12)
    v = np.asarray(vector, dtype=np.float64)
    return v - float(np.dot(v, k)) * k


def estimate_orbit_axis(anchor: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Orbit ekseni k: kamera pozisyonlarinin anchor etrafindaki en iyi duzlem normali.

    k = argmin variance of (C_i - A) along direction (smallest SVD axis).
    """
    rel = np.asarray(positions, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    if len(rel) < 3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    k = vt[2].astype(np.float64)
    unity_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(k, unity_y))) > 0.25:
        if float(np.dot(k, unity_y)) < 0.0:
            k = -k
    norm = float(np.linalg.norm(k))
    return k / norm if norm > 1e-12 else unity_y


def aggregate_mucosa_normal(
    normals: list[np.ndarray] | np.ndarray,
    anchor: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Mukoza dis normali n: crosshair depth patch normal ortalaması (lumen yonune)."""
    if isinstance(normals, np.ndarray):
        n_list = [normals[i] for i in range(len(normals))]
    else:
        n_list = list(normals)
    if not n_list:
        rel = positions - anchor
        fallback = np.mean(rel, axis=0)
        norm = float(np.linalg.norm(fallback))
        return fallback / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    lumen_hint = np.mean(positions - anchor, axis=0)
    lh_norm = float(np.linalg.norm(lumen_hint))
    if lh_norm > 1e-12:
        lumen_hint = lumen_hint / lh_norm

    oriented: list[np.ndarray] = []
    for raw in n_list:
        n = np.asarray(raw, dtype=np.float64)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            continue
        n = n / nn
        if lh_norm > 1e-12 and float(np.dot(n, lumen_hint)) < 0.0:
            n = -n
        oriented.append(n)
    if not oriented:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    n_mean = np.mean(np.stack(oriented, axis=0), axis=0)
    nn = float(np.linalg.norm(n_mean))
    return n_mean / nn if nn > 1e-12 else oriented[0]


def build_geometric_bank_basis(
    anchor: np.ndarray,
    positions: np.ndarray,
    mucosa_normal: np.ndarray,
) -> dict[str, np.ndarray]:
    """Referanssiz orbit bazı: k (orbit ekseni), f (0 deg), r (+90 deg).

    f = normalize(n - (n·k)k)  — mukoza normalinin orbit duzlemine izdusumu
    r = normalize(k x f)
    theta = atan2((C-A)·r, (C-A)·f)
    """
    k = estimate_orbit_axis(anchor, positions)
    n = np.asarray(mucosa_normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)

    f = project_to_plane(n, k)
    fn = float(np.linalg.norm(f))
    if fn < 1e-9:
        rel_mean = np.mean(positions - anchor, axis=0)
        f = project_to_plane(rel_mean, k)
        fn = float(np.linalg.norm(f))
    f = f / max(fn, 1e-12)

    r = np.cross(k, f)
    rn = float(np.linalg.norm(r))
    if rn < 1e-9:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        r = r / rn

    f = np.cross(r, k)
    f = f / max(float(np.linalg.norm(f)), 1e-12)

    return {
        "orbit_axis": k,
        "front": f,
        "right": r,
        "mucosa_normal": n,
    }


def bank_azimuth_geometric(
    cam_pos: np.ndarray,
    anchor: np.ndarray,
    front: np.ndarray,
    right: np.ndarray,
    orbit_axis: np.ndarray,
) -> float:
    """Referanssiz view-bank azimuth (deg): atan2 uzerinde r, f."""
    rel = project_to_plane(
        np.asarray(cam_pos, dtype=np.float64) - np.asarray(anchor, dtype=np.float64),
        orbit_axis,
    )
    x = float(np.dot(rel, right))
    z = float(np.dot(rel, front))
    if np.hypot(x, z) < 1e-9:
        return 0.0
    return wrap_angle_deg(float(np.degrees(np.arctan2(x, z))))


def colab_bank_az_from_geometric(theta_deg: float) -> float:
    """Unity geometric orbit az -> Colab view-bank az (mesh_rotate_y -az)."""
    if not np.isfinite(theta_deg):
        return float("nan")
    return wrap_angle_deg(-float(theta_deg))


def bank_azimuths_geometric(
    anchor: np.ndarray,
    positions: np.ndarray,
    mucosa_normal: np.ndarray,
    *,
    colab_aligned: bool = True,
) -> tuple[np.ndarray, dict]:
    """Tum pozisyonlar icin referanssiz bank azimuth."""
    basis = build_geometric_bank_basis(anchor, positions, mucosa_normal)
    f, r, k = basis["front"], basis["right"], basis["orbit_axis"]
    bank_raw = np.array(
        [bank_azimuth_geometric(p, anchor, f, r, k) for p in positions],
        dtype=np.float64,
    )
    bank = (
        np.array([colab_bank_az_from_geometric(b) for b in bank_raw], dtype=np.float64)
        if colab_aligned
        else bank_raw
    )
    meta = {
        "method": "geometric_mucosa_normal",
        "colab_aligned": bool(colab_aligned),
        "orbit_axis": k.tolist(),
        "front_axis": f.tolist(),
        "right_axis": r.tolist(),
        "mucosa_normal": basis["mucosa_normal"].tolist(),
        "bank_az_raw_min_deg": float(np.min(bank_raw)),
        "bank_az_raw_max_deg": float(np.max(bank_raw)),
        "bank_az_min_deg": float(np.min(bank)),
        "bank_az_max_deg": float(np.max(bank)),
    }
    return bank, meta


def pick_frontal_reference_cam(
    positions: np.ndarray,
    forwards: np.ndarray,
    anchor: np.ndarray,
    *,
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Orbit referansi: forward en iyi anchor'a hizali kare (duz yuz ~ 0 deg)."""
    mask = np.ones(len(positions), dtype=bool) if inlier_mask is None else inlier_mask
    best_idx = None
    best_align = -2.0
    for i in range(len(positions)):
        if not mask[i]:
            continue
        to_a = anchor - positions[i]
        dist = np.linalg.norm(to_a)
        if dist < 1e-8:
            continue
        align = float(np.dot(forwards[i], to_a / dist))
        if align > best_align:
            best_align = align
            best_idx = i
    if best_idx is None:
        return positions[0]
    return positions[int(best_idx)]


def orbit_angle_at_pos(
    cam_pos: np.ndarray,
    anchor: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> float:
    """PCA orbit duzleminde anchor etrafindaki aci (deg)."""
    rel = np.asarray(cam_pos, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    return float(np.degrees(np.arctan2(float(rel @ v_axis), float(rel @ u_axis))))


def pick_reference_frame(
    frame_ids: list[int],
    orbit_theta_deg: np.ndarray,
    *,
    reproj_errors: dict[int, float] | None = None,
    positions: np.ndarray | None = None,
    forwards: np.ndarray | None = None,
    anchor: np.ndarray | None = None,
    method: str = "crosshair_orbit",
    explicit: int | None = None,
    crosshair_max_px: float = 2.0,
) -> int:
    """0 deg referans karesi.

    crosshair_orbit (default): crosshair iyi oturan karelerde orbit acisinin
    medyanina en yakin frame (~ image_0100 bu dataset icin).
    """
    if explicit is not None and int(explicit) in frame_ids:
        return int(explicit)

    if method == "frontal" and positions is not None and forwards is not None and anchor is not None:
        ref_cam = pick_frontal_reference_cam(positions, forwards, anchor)
        for i, p in enumerate(positions):
            if np.allclose(p, ref_cam):
                return int(frame_ids[i])
        return int(frame_ids[0])

    if method == "crosshair_orbit" and reproj_errors:
        good = [
            fid
            for fid in frame_ids
            if np.isfinite(reproj_errors.get(fid, float("nan")))
            and float(reproj_errors[fid]) <= crosshair_max_px
        ]
        if len(good) >= 3:
            good_theta = np.array(
                [float(orbit_theta_deg[frame_ids.index(fid)]) for fid in good],
                dtype=np.float64,
            )
            med = float(np.median(good_theta))
            return int(
                min(good, key=lambda fid: abs(float(orbit_theta_deg[frame_ids.index(fid)]) - med))
            )

    mid = (float(np.min(orbit_theta_deg)) + float(np.max(orbit_theta_deg))) / 2.0
    return int(frame_ids[int(np.argmin(np.abs(orbit_theta_deg - mid)))])


def bank_azimuth_from_orbit(
    cam_pos: np.ndarray,
    anchor: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    ref_orbit_deg: float,
) -> float:
    """view_bank_az: orbit acisi - referans (0 = frontal orbit pozisyonu)."""
    az = orbit_angle_at_pos(cam_pos, anchor, u_axis, v_axis)
    return wrap_angle_deg(az - float(ref_orbit_deg))


def bank_azimuths_for_poses(
    anchor: np.ndarray,
    positions: np.ndarray,
    forwards: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    frame_ids: list[int],
    orbit_theta_deg: np.ndarray,
    *,
    reproj_errors: dict[int, float] | None = None,
    reference_frame: int | None = None,
    reference_method: str = "crosshair_orbit",
) -> tuple[np.ndarray, dict]:
    """Her kamera pozisyonu icin view_bank_az (-90..+90 hedef)."""
    ref_frame = pick_reference_frame(
        frame_ids,
        orbit_theta_deg,
        reproj_errors=reproj_errors,
        positions=positions,
        forwards=forwards,
        anchor=anchor,
        method=reference_method,
        explicit=reference_frame,
    )
    ref_orbit = float(orbit_theta_deg[frame_ids.index(ref_frame)])
    bank = np.array(
        [
            bank_azimuth_from_orbit(p, anchor, u_axis, v_axis, ref_orbit)
            for p in positions
        ],
        dtype=np.float64,
    )
    meta = {
        "reference_frame": int(ref_frame),
        "reference_orbit_deg": ref_orbit,
        "reference_method": reference_method,
        "bank_az_min_deg": float(np.min(bank)),
        "bank_az_max_deg": float(np.max(bank)),
    }
    return bank, meta


def compute_center_ray_gaze(
    anchor: np.ndarray,
    cam_pos: np.ndarray,
    forward: np.ndarray,
) -> tuple[float, float]:
    """Anchor'a bakis hatasi (deg) ve mesafe (m). Ekran merkezi = forward."""
    to_a = anchor - cam_pos
    dist = float(np.linalg.norm(to_a))
    if dist < 1e-8:
        return float("nan"), dist
    err = float(
        np.degrees(
            np.arccos(
                np.clip(float(np.dot(forward, to_a / dist)), -1.0, 1.0)
            )
        )
    )
    return err, dist


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
