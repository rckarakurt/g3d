"""Load polyp PLY meshes exported by polyp_generator (binary or ASCII)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def load_mesh_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vertices (N,3), faces (F,3), normals (N,3), colors (N,3) uint8."""
    path = Path(path)
    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        header = "\n".join(header_lines)
        data = handle.read()

    n_verts = n_faces = 0
    binary = "binary_little_endian" in header
    for line in header_lines:
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])

    verts = np.zeros((n_verts, 3), dtype=np.float64)
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    colors = np.full((n_verts, 3), [204, 106, 118], dtype=np.uint8)

    if binary:
        off = 0
        vert_stride = 27
        for i in range(n_verts):
            x, y, z, nx, ny, nz, r, g, b = struct.unpack_from("<3f3f3B", data, off)
            verts[i] = (x, y, z)
            normals[i] = (nx, ny, nz)
            colors[i] = (r, g, b)
            off += vert_stride
        tris = np.zeros((n_faces, 3), dtype=np.int32)
        for i in range(n_faces):
            _, a, b, c = struct.unpack_from("<B3i", data, off)
            tris[i] = (a, b, c)
            off += 13
    else:
        text = data.decode("utf-8", errors="replace").splitlines()
        idx = 0
        for i in range(n_verts):
            parts = text[idx].split()
            verts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            normals[i] = [float(parts[3]), float(parts[4]), float(parts[5])]
            colors[i] = [int(parts[6]), int(parts[7]), int(parts[8])]
            idx += 1
        tris = np.zeros((n_faces, 3), dtype=np.int32)
        for i in range(n_faces):
            parts = text[idx + i].split()
            tris[i] = [int(parts[1]), int(parts[2]), int(parts[3])]

    return verts, tris, normals, colors


def center_mesh(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move mesh centroid to origin; return centered verts and old center."""
    center = vertices.mean(axis=0)
    return vertices - center, center


def mesh_bounds_radius(vertices: np.ndarray) -> float:
    """Bounding-sphere radius after centering."""
    return float(np.linalg.norm(vertices, axis=1).max())
