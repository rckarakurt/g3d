"""Turntable / wall-mounted render for textured polyp mesh (Colab-safe)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mesh_render import TexturedMeshData, build_textured_mesh, render_mesh_frame_software

# polyp_generator: taban z=0, +Z mukoza disina dogru buyur
WALL_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
WALL_TANGENT_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
MUCOSA_RGB = np.array([188, 108, 118], dtype=np.uint8)


# Unity view_plane_deg (0..180 orbit span) -> bank azimuth (-90..+90, 0=frontal +Z)
UNITY_VIEW_PLANE_OFFSET_DEG = 90.0


def unity_plane_to_bank_az(unity_plane_deg: float) -> float:
    """Unity gaze view_plane 0..180 -> bank -90..+90 (0 = duz yuz bize bakan)."""
    return float(unity_plane_deg) - UNITY_VIEW_PLANE_OFFSET_DEG


def bank_az_to_unity_plane(bank_az_deg: float) -> float:
    return float(bank_az_deg) + UNITY_VIEW_PLANE_OFFSET_DEG


def wall_azimuth_grid(step_deg: int = 5, half_span_deg: int = 90) -> list[float]:
    """Dikey (Y) eksen etrafinda -half_span .. +half_span; 0 = on (+Z, duz yuz)."""
    step = max(1, int(step_deg))
    span = min(90, max(step, int(half_span_deg)))
    return [float(a) for a in range(-span, span + 1, step)]


def preview_azimuth_grid(step_deg: int = 15, half_span_deg: int = 90) -> list[float]:
    """Onizleme: -90 .. +90."""
    return wall_azimuth_grid(step_deg, half_span_deg)


def azimuth_view_filename(azimuth_deg: float) -> str:
    """Signed azimuth -> view_azm045.png / view_azp000.png."""
    n = int(round(float(azimuth_deg)))
    if n < 0:
        return f"view_azm{abs(n):03d}.png"
    return f"view_azp{n:03d}.png"


DEFAULT_WALL_AZIMUTH_STEP_DEG = 5
DEFAULT_WALL_AZIMUTH_HALF_SPAN_DEG = 90
DEFAULT_WALL_AZIMUTHS_DEG = wall_azimuth_grid(
    DEFAULT_WALL_AZIMUTH_STEP_DEG, DEFAULT_WALL_AZIMUTH_HALF_SPAN_DEG
)


def pinhole_intrinsics(
    width: int,
    height: int,
    *,
    fov_deg: float = 45.0,
) -> np.ndarray:
    """OpenCV-style K matrix."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def rotate_vertices_y(
    vertices: np.ndarray,
    angle_deg: float,
    center: np.ndarray,
) -> np.ndarray:
    """Y ekseni (dikey) etrafinda dondur — duvara yapisik polyp turntable."""
    a = np.radians(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    rel = vertices.astype(np.float64) - center
    return (rel @ R.T) + center


def _camera_c2w_opencv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Roll-stabil look-at (OpenCV: +X sag, +Y asagi, +Z ileri)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z_fwd = target - eye
    z_fwd = z_fwd / (np.linalg.norm(z_fwd) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_right = np.cross(z_fwd, world_up)
    if np.linalg.norm(x_right) < 1e-8:
        x_right = np.array([1.0, 0.0, 0.0])
    else:
        x_right = x_right / np.linalg.norm(x_right)
    y_down = np.cross(z_fwd, x_right)
    y_down = y_down / (np.linalg.norm(y_down) + 1e-12)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x_right
    c2w[:3, 1] = y_down
    c2w[:3, 2] = z_fwd
    c2w[:3, 3] = eye
    return c2w


def lumen_camera_c2w(
    distance: float,
    target: np.ndarray,
    *,
    obliquity_deg: float = 18.0,
) -> np.ndarray:
    """Sabit kamera: +Z lumen (0 derece = duz yuz). Mesh Y ekseninde doner."""
    t = np.asarray(target, dtype=np.float64)
    obl = np.radians(float(obliquity_deg))
    eye = t + distance * np.array([0.0, np.sin(obl), np.cos(obl)], dtype=np.float64)
    return _camera_c2w_opencv(eye, t)


def look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Camera-to-world (OpenCV: +X right, +Y down, +Z forward)."""
    up = WALL_TANGENT_UP if up is None else np.asarray(up, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)

    down = np.cross(forward, right)
    down = down / (np.linalg.norm(down) + 1e-12)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def align_polyp_to_wall(
    vertices: np.ndarray,
    normal: np.ndarray | None = None,
) -> np.ndarray:
    """Taban duvar duzlemine otursun (min projection -> 0)."""
    n = np.asarray(normal if normal is not None else WALL_NORMAL, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    verts = vertices.astype(np.float64).copy()
    base = float(np.min(verts @ n))
    verts -= base * n
    return verts


def polyp_gaze_target(vertices: np.ndarray, normal: np.ndarray | None = None) -> np.ndarray:
    """Bakis hedefi: lezyon govdesi (taban degil)."""
    n = np.asarray(normal if normal is not None else WALL_NORMAL, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    h = vertices @ n
    lo, hi = np.percentile(h, [25, 90])
    mask = (h >= lo) & (h <= hi)
    if mask.sum() < 3:
        return vertices.mean(axis=0)
    return vertices[mask].mean(axis=0)


def wall_orbit_camera(
    azimuth_deg: float,
    *,
    obliquity_deg: float = 18.0,
    distance: float,
    target: np.ndarray | None = None,
    orbit_mode: str = "lumen",
) -> np.ndarray:
    """Mukoza duvarina yapisik polyp kamerasi.

    orbit_mode:
      - ``lumen`` (varsayilan): **Y ekseni** (dikey), -90..+90°.
        0 = duz yuz bize bakan (+Z lumen), -90 = sol yan, +90 = sag yan.
      - ``full``: tam 360° (debug).
    """
    target = np.zeros(3, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
    obl = np.radians(obliquity_deg)

    if orbit_mode == "full":
        az = np.radians(float(azimuth_deg) % 360.0)
        direction = np.array(
            [
                np.sin(obl) * np.cos(az),
                np.sin(obl) * np.sin(az),
                np.cos(obl) * np.cos(az),
            ],
            dtype=np.float64,
        )
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        eye = target + distance * direction
        return _camera_c2w_opencv(eye, target)
    else:
        az = np.radians(float(np.clip(azimuth_deg, -90.0, 90.0)))
        eye = target + distance * np.array(
            [np.sin(az), 0.0, np.cos(az)],
            dtype=np.float64,
        )
        return _camera_c2w_opencv(eye, target)


def turntable_camera(
    azimuth_deg: float,
    *,
    elevation_deg: float = 25.0,
    distance: float,
    target: np.ndarray | None = None,
) -> np.ndarray:
    """Legacy sphere orbit — prefer wall_orbit_camera for endoscopy."""
    target = np.zeros(3) if target is None else np.asarray(target, dtype=np.float64)
    return wall_orbit_camera(
        azimuth_deg,
        obliquity_deg=elevation_deg,
        distance=distance,
        target=target,
    )


def make_wall_disk(
    center: np.ndarray,
    *,
    radius: float,
    z: float | None = None,
    segments: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """XY duvar disk (mukoza), hafif z ofset."""
    center = np.asarray(center, dtype=np.float64)
    z_use = float(center[2] if z is None else z) - 0.05
    verts = [[center[0], center[1], z_use]]
    faces = []
    for i in range(segments):
        a0 = 2.0 * np.pi * i / segments
        a1 = 2.0 * np.pi * (i + 1) / segments
        verts.append(
            [
                center[0] + radius * np.cos(a0),
                center[1] + radius * np.sin(a0),
                z_use,
            ]
        )
        verts.append(
            [
                center[0] + radius * np.cos(a1),
                center[1] + radius * np.sin(a1),
                z_use,
            ]
        )
        faces.append([0, 1 + i * 2, 2 + i * 2])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _rasterize_flat_tri(
    pts: np.ndarray,
    z_tri: np.ndarray,
    color: np.ndarray,
    out: np.ndarray,
    zbuf: np.ndarray,
    min_u: int,
    max_u: int,
    min_v: int,
    max_v: int,
) -> None:
    v0, v1, v2 = pts[0], pts[1], pts[2]
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-12:
        return

    uu = np.arange(min_u, max_u + 1, dtype=np.float64) + 0.5
    vv = np.arange(min_v, max_v + 1, dtype=np.float64) + 0.5
    pu, pv = np.meshgrid(uu, vv, indexing="xy")

    w0 = ((v1[1] - v2[1]) * (pu - v2[0]) + (v2[0] - v1[0]) * (pv - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (pu - v2[0]) + (v0[0] - v2[0]) * (pv - v2[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    if not inside.any():
        return

    z = w0 * z_tri[0] + w1 * z_tri[1] + w2 * z_tri[2]
    valid = inside & (z > 1e-4)
    if not valid.any():
        return

    u_idx = np.arange(min_u, max_u + 1)
    v_idx = np.arange(min_v, max_v + 1)
    uu_i, vv_i = np.meshgrid(u_idx, v_idx, indexing="xy")

    zb = zbuf[vv_i, uu_i]
    closer = valid & (z < zb)
    if not closer.any():
        return

    zbuf[vv_i, uu_i] = np.where(closer, z.astype(np.float32), zb)
    for ch in range(3):
        plane = out[:, :, ch]
        plane[vv_i, uu_i] = np.where(closer, int(color[ch]), plane[vv_i, uu_i])
        out[:, :, ch] = plane


def _project_tris(
    vertices: np.ndarray,
    faces: np.ndarray,
    w2c: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hom = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float64)])
    cam = (w2c @ hom.T).T[:, :3]
    z = cam[:, 2]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    zsafe = np.maximum(z, 1e-6)
    us = fx * cam[:, 0] / zsafe + cx
    vs = fy * cam[:, 1] / zsafe + cy
    return us, vs, z


def render_flat_mesh_software(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: np.ndarray,
    k: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    out: np.ndarray,
    zbuf: np.ndarray,
) -> None:
    w2c = np.linalg.inv(c2w.astype(np.float64))
    us, vs, zv = _project_tris(vertices, faces, w2c, k)
    color = np.asarray(color, dtype=np.uint8)

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if zv[i0] <= 1e-4 or zv[i1] <= 1e-4 or zv[i2] <= 1e-4:
            continue
        pts = np.array([[us[i0], vs[i0]], [us[i1], vs[i1]], [us[i2], vs[i2]]])
        min_u = max(0, int(np.floor(pts[:, 0].min())))
        max_u = min(width - 1, int(np.ceil(pts[:, 0].max())))
        min_v = max(0, int(np.floor(pts[:, 1].min())))
        max_v = min(height - 1, int(np.ceil(pts[:, 1].max())))
        if min_u > max_u or min_v > max_v:
            continue
        _rasterize_flat_tri(
            pts,
            np.array([zv[i0], zv[i1], zv[i2]], dtype=np.float64),
            color,
            out,
            zbuf,
            min_u,
            max_u,
            min_v,
            max_v,
        )


def render_wall_mounted_frame(
    data: TexturedMeshData,
    k: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    *,
    wall_center: np.ndarray,
    wall_radius: float,
    mucosa_rgb: np.ndarray | None = None,
    include_mucosa: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Textured polyp; opsiyonel mukoza disk (onizleme icin).

    View bank / composite icin include_mucosa=False — yalnizca polyp, seffaf arka plan.
    """
    out = np.zeros((height, width, 3), dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    if include_mucosa:
        mucosa = MUCOSA_RGB if mucosa_rgb is None else np.asarray(mucosa_rgb, dtype=np.uint8)
        w_verts, w_faces = make_wall_disk(wall_center, radius=wall_radius)
        render_flat_mesh_software(w_verts, w_faces, mucosa, k, c2w, width, height, out, zbuf)

    w2c = np.linalg.inv(c2w.astype(np.float64))
    hom = np.hstack([data.vertices, np.ones((len(data.vertices), 1), dtype=np.float64)])
    cam = (w2c @ hom.T).T[:, :3]
    zv = cam[:, 2]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    zsafe = np.maximum(zv, 1e-6)
    us = fx * cam[:, 0] / zsafe + cx
    vs = fy * cam[:, 1] / zsafe + cy
    tex = data.texture

    from mesh_render import _rasterize_tri_vectorized

    for tri in data.faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if zv[i0] <= 1e-4 or zv[i1] <= 1e-4 or zv[i2] <= 1e-4:
            continue
        pts = np.array([[us[i0], vs[i0]], [us[i1], vs[i1]], [us[i2], vs[i2]]])
        min_u = max(0, int(np.floor(pts[:, 0].min())))
        max_u = min(width - 1, int(np.ceil(pts[:, 0].max())))
        min_v = max(0, int(np.floor(pts[:, 1].min())))
        max_v = min(height - 1, int(np.ceil(pts[:, 1].max())))
        if min_u > max_u or min_v > max_v:
            continue
        _rasterize_tri_vectorized(
            pts,
            data.uvs[[i0, i1, i2]],
            np.array([zv[i0], zv[i1], zv[i2]], dtype=np.float64),
            tex,
            out,
            zbuf,
            min_u,
            max_u,
            min_v,
            max_v,
        )
    return out, zbuf


def _render_polyp_rgb_zbuf(
    data: TexturedMeshData,
    k: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Polyp mesh only — no mucosa disk."""
    w2c = np.linalg.inv(c2w.astype(np.float64))
    hom = np.hstack([data.vertices, np.ones((len(data.vertices), 1), dtype=np.float64)])
    cam = (w2c @ hom.T).T[:, :3]
    zv = cam[:, 2]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    zsafe = np.maximum(zv, 1e-6)
    us = fx * cam[:, 0] / zsafe + cx
    vs = fy * cam[:, 1] / zsafe + cy
    tex = data.texture

    out = np.zeros((height, width, 3), dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    from mesh_render import _rasterize_tri_vectorized

    for tri in data.faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if zv[i0] <= 1e-4 or zv[i1] <= 1e-4 or zv[i2] <= 1e-4:
            continue
        pts = np.array([[us[i0], vs[i0]], [us[i1], vs[i1]], [us[i2], vs[i2]]])
        min_u = max(0, int(np.floor(pts[:, 0].min())))
        max_u = min(width - 1, int(np.ceil(pts[:, 0].max())))
        min_v = max(0, int(np.floor(pts[:, 1].min())))
        max_v = min(height - 1, int(np.ceil(pts[:, 1].max())))
        if min_u > max_u or min_v > max_v:
            continue
        _rasterize_tri_vectorized(
            pts,
            data.uvs[[i0, i1, i2]],
            np.array([zv[i0], zv[i1], zv[i2]], dtype=np.float64),
            tex,
            out,
            zbuf,
            min_u,
            max_u,
            min_v,
            max_v,
        )
    return out, zbuf


def render_mesh_rgba_software(
    data: TexturedMeshData,
    k: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    *,
    wall_mounted: bool = True,
    wall_center: np.ndarray | None = None,
    wall_radius: float | None = None,
    include_mucosa: bool = False,
) -> np.ndarray:
    """RGBA PNG: polyp opak, arka plan seffaf (z-buffer alpha)."""
    if wall_mounted:
        center = wall_center if wall_center is not None else polyp_gaze_target(data.vertices)
        radius = wall_radius if wall_radius is not None else float(np.linalg.norm(data.vertices, axis=1).max()) * 2.2
        rgb, zbuf = render_wall_mounted_frame(
            data,
            k,
            c2w,
            width,
            height,
            wall_center=center,
            wall_radius=radius,
            include_mucosa=include_mucosa,
        )
    else:
        rgb, zbuf = _render_polyp_rgb_zbuf(data, k, c2w, width, height)

    alpha = np.where(zbuf < np.inf, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def render_turntable_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_rgb: np.ndarray,
    azimuths_deg: list[float] | tuple[float, ...],
    out_dir: Path,
    *,
    width: int = 768,
    height: int = 768,
    elevation_deg: float = 18.0,
    distance_scale: float = 2.8,
    simplify: bool = True,
    wall_mounted: bool = True,
    orbit_mode: str = "lumen",
    include_mucosa: bool = False,
) -> dict:
    """Render textured polyp from wall-orbit azimuths; save RGBA PNG + manifest."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts = align_polyp_to_wall(vertices) if wall_mounted else vertices.astype(np.float64)
    target = polyp_gaze_target(verts)
    radius = float(np.linalg.norm(verts, axis=1).max())
    distance = max(radius * distance_scale, 1e-3)
    k = pinhole_intrinsics(width, height, fov_deg=50.0)
    wall_center = np.array([target[0], target[1], 0.0], dtype=np.float64)

    use_mesh_rotate = wall_mounted and orbit_mode == "lumen"
    c2w_fixed = (
        lumen_camera_c2w(distance, target, obliquity_deg=elevation_deg)
        if use_mesh_rotate
        else None
    )

    entries = []
    for az in azimuths_deg:
        az_f = float(az)
        if use_mesh_rotate:
            verts_az = rotate_vertices_y(verts, -az_f, target)
            c2w = c2w_fixed
        else:
            verts_az = verts
            c2w = wall_orbit_camera(
                az_f,
                obliquity_deg=elevation_deg,
                distance=distance,
                target=target,
                orbit_mode=orbit_mode,
            )
        mesh_data = build_textured_mesh(
            verts_az, faces, uvs, texture_rgb, simplify=simplify
        )
        rgba = render_mesh_rgba_software(
            mesh_data,
            k,
            c2w,
            width,
            height,
            wall_mounted=wall_mounted,
            wall_center=wall_center,
            wall_radius=radius * 2.4,
            include_mucosa=include_mucosa,
        )
        name = azimuth_view_filename(az)
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        entries.append(
            {
                "azimuth_deg": float(az),
                "obliquity_deg": float(elevation_deg),
                "file": name,
                "width": width,
                "height": height,
            }
        )

    manifest = {
        "render_mode": "wall_mounted" if wall_mounted else "turntable",
        "orbit_mode": orbit_mode,
        "turntable_method": "mesh_rotate_y" if use_mesh_rotate else "orbit_camera",
        "wall_normal": WALL_NORMAL.tolist(),
        "azimuths_deg": [float(a) for a in azimuths_deg],
        "obliquity_deg": float(elevation_deg),
        "camera_distance_mm": distance,
        "gaze_target_mm": target.tolist(),
        "views": entries,
    }
    manifest_path = out_dir / "view_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_rgba(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        gray = img
        return np.dstack([gray, gray, gray, np.full_like(gray, 255)])
    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        return np.dstack([rgb, alpha])
    bgra = img
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    alpha = bgra[:, :, 3]
    return np.dstack([rgb, alpha]).astype(np.uint8)


def rgba_on_background(rgba: np.ndarray, bg_rgb: tuple[int, int, int] = (12, 12, 16)) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.array(bg_rgb, dtype=np.float32)
    out = rgb * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_view_strip(
    view_dir: Path,
    manifest: dict | None = None,
    *,
    thumb_height: int = 320,
    gap_px: int = 10,
    bg_rgb: tuple[int, int, int] = (12, 12, 16),
    label_color: tuple[int, int, int] = (240, 240, 240),
    max_cols: int | None = None,
) -> np.ndarray:
    import cv2

    view_dir = Path(view_dir)
    if manifest is None:
        manifest_path = view_dir / "view_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest yok: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    panels: list[np.ndarray] = []
    for entry in manifest["views"]:
        path = view_dir / entry["file"]
        rgba = _load_rgba(path)
        rgb = rgba_on_background(rgba, bg_rgb=bg_rgb)
        h, w = rgb.shape[:2]
        scale = thumb_height / max(h, 1)
        thumb_w = max(1, int(round(w * scale)))
        thumb = cv2.resize(rgb, (thumb_w, thumb_height), interpolation=cv2.INTER_AREA)

        label_h = 36
        band = np.full((label_h, thumb_w, 3), bg_rgb, dtype=np.uint8)
        text = f"{int(round(entry['azimuth_deg'])):+d}\u00b0"
        cv2.putText(
            band,
            text,
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            label_color,
            2,
            cv2.LINE_AA,
        )
        panels.append(np.vstack([band, thumb]))

    if not panels:
        raise RuntimeError("Gosterilecek view yok.")

    if max_cols is None or len(panels) <= max_cols:
        strip = panels[0]
        for panel in panels[1:]:
            pad = np.full((strip.shape[0], gap_px, 3), bg_rgb, dtype=np.uint8)
            strip = np.hstack([strip, pad, panel])
        return strip

    # Grid: equal-width columns per row
    ncol = max(1, int(max_cols))
    target_w = max(p.shape[1] for p in panels)
    norm: list[np.ndarray] = []
    for p in panels:
        h, w = p.shape[:2]
        if w != target_w:
            scale = target_w / max(w, 1)
            new_h = max(1, int(round(h * scale)))
            p = cv2.resize(p, (target_w, new_h), interpolation=cv2.INTER_AREA)
        norm.append(p)

    row_h = max(p.shape[0] for p in norm)
    rows: list[np.ndarray] = []
    vgap = np.full((gap_px, target_w, 3), bg_rgb, dtype=np.uint8)
    for i in range(0, len(norm), ncol):
        chunk = norm[i : i + ncol]
        while len(chunk) < ncol:
            pad_panel = np.full((row_h, target_w, 3), bg_rgb, dtype=np.uint8)
            chunk.append(pad_panel)
        row_parts: list[np.ndarray] = []
        for j, panel in enumerate(chunk):
            if panel.shape[0] < row_h:
                pad = np.full((row_h - panel.shape[0], target_w, 3), bg_rgb, dtype=np.uint8)
                panel = np.vstack([panel, pad])
            row_parts.append(panel)
            if j + 1 < ncol:
                row_parts.append(np.full((row_h, gap_px, 3), bg_rgb, dtype=np.uint8))
        rows.append(np.hstack(row_parts))

    grid = rows[0]
    for row in rows[1:]:
        hgap = np.full((gap_px, grid.shape[1], 3), bg_rgb, dtype=np.uint8)
        grid = np.vstack([grid, hgap, row])
    return grid


def save_view_strip(
    view_dir: Path,
    out_path: Path | None = None,
    **kwargs,
) -> Path:
    import cv2

    view_dir = Path(view_dir)
    strip = build_view_strip(view_dir, **kwargs)
    out_path = Path(out_path or view_dir / "view_strip.png")
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return out_path


def show_view_strip_colab(
    view_dir: Path,
    *,
    figsize_per_view: float = 1.1,
    thumb_height: int = 260,
    max_cols: int | None = 7,
    save: bool = True,
    strip_name: str = "view_strip.png",
) -> np.ndarray:
    import matplotlib.pyplot as plt

    view_dir = Path(view_dir)
    strip = build_view_strip(view_dir, thumb_height=thumb_height, max_cols=max_cols)
    if save:
        import cv2

        out_path = view_dir / strip_name
        cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    manifest = json.loads((view_dir / "view_manifest.json").read_text(encoding="utf-8"))
    n_views = len(manifest["views"])
    orbit_mode = manifest.get("orbit_mode", "lumen")
    fig_w = max(min(figsize_per_view * min(n_views, max_cols or n_views), 18.0), 8.0)
    n_rows = 1 if max_cols is None else int(np.ceil(n_views / max(max_cols, 1)))
    fig_h = max(3.5, 2.2 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(strip)
    ax.axis("off")
    ax.set_title(
        f"Polyp on mucosa — azimuth preview ({orbit_mode} orbit)",
        fontsize=13,
        pad=10,
    )
    plt.tight_layout()
    plt.show()
    return strip
