"""Textured mesh render — pyrender (EGL) with software fallback for Colab."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def setup_colab_gl() -> None:
    """Call once before any pyrender import (Bölüm 9 başı)."""
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("EGL_DEVICE_ID", "0")
    try:
        import subprocess
        subprocess.run(
            ["apt-get", "install", "-qq", "-y",
             "libegl1-mesa", "libgl1-mesa-glx", "libosmesa6"],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass


@dataclass
class TexturedMeshData:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    texture: np.ndarray


def simplify_mesh_for_render(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    *,
    target_tris: int = 12000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce triangle count; remap UVs to simplified vertices."""
    if len(faces) <= target_tris:
        return vertices, faces, uvs
    try:
        import open3d as o3d
        from scipy.spatial import cKDTree

        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32)),
        )
        m.remove_degenerate_triangles()
        m = m.simplify_quadric_decimation(max(int(target_tris), 1000))
        v_new = np.asarray(m.vertices, dtype=np.float64)
        f_new = np.asarray(m.triangles, dtype=np.int32)
        _, nn = cKDTree(vertices.astype(np.float64)).query(v_new, k=1)
        uv_new = uvs[nn].astype(np.float32)
        return v_new, f_new, uv_new
    except Exception as exc:
        print("Mesh simplify atlandi:", exc)
        return vertices, faces, uvs


def build_textured_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_rgb: np.ndarray,
    *,
    simplify: bool = True,
) -> TexturedMeshData:
    v, f, uv = (
        simplify_mesh_for_render(vertices, faces, uvs)
        if simplify
        else (vertices, faces, uvs)
    )
    return TexturedMeshData(
        vertices=v.astype(np.float64),
        faces=f.astype(np.int32),
        uvs=uv.astype(np.float32),
        texture=np.ascontiguousarray(texture_rgb.astype(np.uint8)),
    )


# Legacy alias
def build_textured_trimesh(vertices, faces, uvs, texture_rgb):
    data = build_textured_mesh(vertices, faces, uvs, texture_rgb)
    from PIL import Image
    import trimesh
    from trimesh.visual.texture import TextureVisuals

    img = Image.fromarray(data.texture)
    return trimesh.Trimesh(
        vertices=data.vertices,
        faces=data.faces,
        visual=TextureVisuals(uv=data.uvs, image=img),
        process=False,
    )


def _project_world_to_pixel(
    vertices: np.ndarray, w2c: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hom = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float64)])
    cam = (w2c @ hom.T).T[:, :3]
    z = cam[:, 2]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    zsafe = np.maximum(z, 1e-6)
    us = fx * cam[:, 0] / zsafe + cx
    vs = fy * cam[:, 1] / zsafe + cy
    return us, vs, z


def _rasterize_tri_vectorized(
    pts: np.ndarray,
    uv_tri: np.ndarray,
    z_tri: np.ndarray,
    tex: np.ndarray,
    out: np.ndarray,
    zbuf: np.ndarray,
    min_u: int,
    max_u: int,
    min_v: int,
    max_v: int,
) -> None:
    """Vectorized barycentric raster inside triangle bbox."""
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

    uvs = (
        w0[..., None] * uv_tri[0]
        + w1[..., None] * uv_tri[1]
        + w2[..., None] * uv_tri[2]
    )
    th, tw = tex.shape[:2]
    tu = np.clip((uvs[..., 0] * (tw - 1)).astype(np.int32), 0, tw - 1)
    tv = np.clip(((1.0 - uvs[..., 1]) * (th - 1)).astype(np.int32), 0, th - 1)
    colors = tex[tv, tu]

    zbuf[vv_i, uu_i] = np.where(closer, z.astype(np.float32), zb)
    for ch in range(3):
        out_ch = out[:, :, ch]
        out_ch[vv_i, uu_i] = np.where(closer, colors[..., ch], out_ch[vv_i, uu_i])
        out[:, :, ch] = out_ch


def render_mesh_frame_software(
    data: TexturedMeshData,
    k: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Pure numpy textured raster — Colab'da pyrender olmadan calisir."""
    w2c = np.linalg.inv(c2w.astype(np.float64))
    us, vs, zv = _project_world_to_pixel(data.vertices, w2c, k)

    out = np.zeros((height, width, 3), dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    tex = data.texture

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
    return out


def _render_pyrender(tmesh, k, pyrender_pose, width, height, ambient_light=0.55):
    setup_colab_gl()
    import pyrender

    mesh_pr = pyrender.Mesh.from_trimesh(tmesh, smooth=True)
    scene = pyrender.Scene(ambient_light=[ambient_light] * 3, bg_color=[0, 0, 0, 0])
    scene.add(mesh_pr)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=2.0
    )
    cam_node = scene.add(camera, pose=pyrender_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
    scene.add(light, pose=pyrender_pose)
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return color[:, :, :3].astype(np.uint8)


def render_mesh_frame(
    mesh_or_data,
    k: np.ndarray,
    pyrender_pose: np.ndarray,
    width: int,
    height: int,
    *,
    ambient_light: float = 0.55,
    c2w: np.ndarray | None = None,
    prefer_software: bool = True,
) -> np.ndarray:
    """Render textured mesh. Default: software (Colab-safe)."""
    if isinstance(mesh_or_data, TexturedMeshData):
        data = mesh_or_data
    else:
        # trimesh legacy
        tm = mesh_or_data
        data = TexturedMeshData(
            vertices=np.asarray(tm.vertices),
            faces=np.asarray(tm.faces),
            uvs=np.asarray(tm.visual.uv),
            texture=np.array(tm.visual.material.image),
        )

    c2w_use = c2w if c2w is not None else pyrender_pose  # caller should pass c2w

    if prefer_software:
        return render_mesh_frame_software(data, k, c2w_use, width, height)

    try:
        import trimesh
        from trimesh.visual.texture import TextureVisuals
        from PIL import Image

        img = Image.fromarray(data.texture)
        tm = trimesh.Trimesh(
            vertices=data.vertices,
            faces=data.faces,
            visual=TextureVisuals(uv=data.uvs, image=img),
            process=False,
        )
        return _render_pyrender(tm, k, pyrender_pose, width, height, ambient_light)
    except Exception as exc:
        print("pyrender basarisiz, software fallback:", exc)
        return render_mesh_frame_software(data, k, c2w_use, width, height)


def composite_rendered_polyp(
    unity_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    polyp_mask: np.ndarray,
    *,
    use_poisson: bool = True,
    lab_match: bool = True,
    lab_strength: float = 0.45,
) -> np.ndarray:
    from coating_utils import blend_polyp

    coat = polyp_mask if polyp_mask.max() else (rendered_rgb.sum(axis=2) > 10).astype(np.uint8) * 255
    if coat.ndim == 3:
        coat = coat[:, :, 0]
    return blend_polyp(
        unity_rgb, rendered_rgb, coat,
        use_poisson=use_poisson, lab_match=lab_match, lab_strength=lab_strength,
    )
