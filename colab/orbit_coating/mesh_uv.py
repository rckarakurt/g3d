"""UV unwrap (xatlas) + texture inpaint for polyp mesh."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def unwrap_mesh_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    atlas_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xatlas parametrization → UVs aligned with remapped vertices.

    Returns (vertices_remapped, faces_remapped, uvs) where uvs is (N_verts, 2).
    """
    import xatlas

    vmapping, faces_remapped, uvs = xatlas.parametrize(
        vertices.astype(np.float32),
        faces.astype(np.int32),
    )
    verts_out = vertices[vmapping]
    uvs = uvs.astype(np.float32)
    return verts_out, faces_remapped.astype(np.int32), uvs


def build_uv_mesh_data(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
) -> dict:
    """Pack mesh + UV for baking/render."""
    return {
        "vertices": vertices.astype(np.float64),
        "faces": faces.astype(np.int32),
        "uvs": uvs.astype(np.float32),
    }


def inpaint_uv_texture(
    texture: np.ndarray,
    weight: np.ndarray,
    *,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """Fill UV texels never seen by any canonical view."""
    tex = texture.copy()
    w = weight.copy()
    holes = (w <= 1e-6).astype(np.uint8) * 255
    if holes.max() == 0:
        return tex
    filled = cv2.inpaint(tex, holes, inpaint_radius, cv2.INPAINT_TELEA)
    valid = w > 1e-6
    out = filled.copy()
    out[valid] = tex[valid]
    return out


def normalize_texture(
    rgb_acc: np.ndarray,
    weight: np.ndarray,
    *,
    default_rgb: tuple[int, int, int] = (160, 100, 90),
) -> np.ndarray:
    """Weighted average → uint8 RGB."""
    w = np.maximum(weight, 0.0)
    out = np.zeros_like(rgb_acc, dtype=np.float32)
    valid = w > 1e-6
    out[valid] = rgb_acc[valid] / w[valid, None]
    if not valid.any():
        out[:] = np.array(default_rgb, dtype=np.float32)
    else:
        fill = out[valid].mean(axis=0)
        out[~valid] = fill
    return np.clip(out, 0, 255).astype(np.uint8)


def save_uv_preview(
    texture: np.ndarray,
    path: Path,
    *,
    grid: int = 16,
) -> None:
    """Save UV atlas with optional grid overlay for debugging."""
    vis = texture.copy()
    h, w = vis.shape[:2]
    for i in range(0, w, max(w // grid, 1)):
        vis[:, i : i + 1] = (vis[:, i : i + 1].astype(np.int32) + [40, 40, 40]) // 2
    for j in range(0, h, max(h // grid, 1)):
        vis[j : j + 1, :] = (vis[j : j + 1, :].astype(np.int32) + [40, 40, 40]) // 2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
