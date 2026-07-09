"""StyleShot UV texture generation for single PLY (encoder via gdown, no Drive upload)."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from styleshot_helpers import build_control_image, generate_syn, init_styleshot

STYLE_REF_CACHE = Path("/content/vrcaps_checkpoints/kvasir_style_ref.jpg")
KVASIR_ZIP_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_ZIP = Path("/content/kvasir-seg.zip")

DEFAULT_PROMPT = (
    "colonoscopy image of a sessile polyp on pink mucosa, "
    "smooth organic surface, realistic endoscopic texture, clinical photography"
)


def _use_kvasir_download() -> bool:
    """Colab SSL sorunlari icin varsayilan: Kvasir indirme KAPALI."""
    return os.environ.get("VRCAPS_USE_KVASIR", "0").strip() == "1"


def _synthetic_mucosa_style_ref(dest: Path, size: int = 512) -> Path:
    """SSL / indirme basarisiz olursa: pembe mukoza benzeri sentetik stil ref."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    base = np.array([188, 108, 118], dtype=np.float32)
    img = base + rng.normal(0, 22, (size, size, 3))
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    vignette = 1.0 - 0.35 * np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    img *= vignette[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(dest), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("Sentetik stil referansi kullaniliyor:", dest)
    return dest


def ensure_style_ref(dest: Path | None = None) -> Path:
    """Stil referansi: Drive/local varsa onu kullan; yoksa sentetik (ag yok)."""
    dest = Path(dest or STYLE_REF_CACHE)
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest

    drive_candidates = [
        Path("/content/drive/MyDrive/vrcaps/kvasir_style_ref.jpg"),
        Path("/content/drive/MyDrive/vrcaps/kvasir_style_ref.png"),
    ]
    for candidate in drive_candidates:
        if candidate.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dest)
            return dest

    k512 = Path("/content/StyleShot/data/kvasir/images_512")
    if k512.exists():
        imgs = sorted(k512.glob("*.jpg")) + sorted(k512.glob("*.png"))
        if imgs:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(imgs[0], dest)
            return dest

    if not _use_kvasir_download():
        return _synthetic_mucosa_style_ref(dest)

    from colab_net import configure_colab_ssl, download_url

    configure_colab_ssl()
    try:
        if not KVASIR_ZIP.exists() or KVASIR_ZIP.stat().st_size < 1_000_000:
            print("Kvasir stil referansi indiriliyor (~44 MB)...")
            download_url(KVASIR_ZIP_URL, KVASIR_ZIP)

        tmp = Path("/content/_kvasir_ref_tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(KVASIR_ZIP, "r") as archive:
            archive.extractall(tmp)

        imgs = sorted(tmp.rglob("*.jpg")) + sorted(tmp.rglob("*.png"))
        if not imgs:
            raise FileNotFoundError("Kvasir zip icinde gorsel yok")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imgs[0], dest)
        return dest
    except Exception as exc:
        print("Kvasir indirme/acma basarisiz:", exc)
        return _synthetic_mucosa_style_ref(dest)


def _project_vertices(
    vertices: np.ndarray,
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


def render_vertex_color(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    k: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Software raster with per-vertex RGB; returns rgb, mask, depth."""
    w2c = np.linalg.inv(c2w.astype(np.float64))
    us, vs, zv = _project_vertices(vertices, w2c, k)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

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

        z_tri = np.array([zv[i0], zv[i1], zv[i2]], dtype=np.float64)
        col_tri = colors[[i0, i1, i2]].astype(np.float32)
        v0, v1, v2 = pts[0], pts[1], pts[2]
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-12:
            continue

        uu = np.arange(min_u, max_u + 1, dtype=np.float64) + 0.5
        vv = np.arange(min_v, max_v + 1, dtype=np.float64) + 0.5
        pu, pv = np.meshgrid(uu, vv, indexing="xy")
        w0 = ((v1[1] - v2[1]) * (pu - v2[0]) + (v2[0] - v1[0]) * (pv - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (pu - v2[0]) + (v0[0] - v2[0]) * (pv - v2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        z = w0 * z_tri[0] + w1 * z_tri[1] + w2 * z_tri[2]
        valid = inside & (z > 1e-4)
        if not valid.any():
            continue

        u_idx = np.arange(min_u, max_u + 1)
        v_idx = np.arange(min_v, max_v + 1)
        uu_i, vv_i = np.meshgrid(u_idx, v_idx, indexing="xy")
        zb = zbuf[vv_i, uu_i]
        closer = valid & (z < zb)
        if not closer.any():
            continue

        cols = (
            w0[..., None] * col_tri[0]
            + w1[..., None] * col_tri[1]
            + w2[..., None] * col_tri[2]
        )
        zbuf[vv_i, uu_i] = np.where(closer, z.astype(np.float32), zb)
        depth[vv_i, uu_i] = np.where(closer, z.astype(np.float32), depth[vv_i, uu_i])
        for ch in range(3):
            plane = rgb[:, :, ch]
            plane[vv_i, uu_i] = np.where(closer, cols[..., ch].astype(np.uint8), plane[vv_i, uu_i])
            rgb[:, :, ch] = plane

    mask = (zbuf < np.inf).astype(np.uint8) * 255
    return rgb, mask, depth


def generate_uv_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    out_size: int = 2048,
    seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
    controlnet_scale: float = 0.55,
    use_content_encoder: bool = False,
    control_size: int = 512,
    elevation_deg: float = 18.0,
) -> tuple[np.ndarray, dict]:
    """StyleShot ile UV atlas dokusu uret."""
    from ply_loader import mesh_bounds_radius
    from turntable_render import align_polyp_to_wall, pinhole_intrinsics, polyp_gaze_target, wall_orbit_camera

    style_ref = ensure_style_ref()
    styleshot, detector, content_ckpt, style_ckpt, _pipe = init_styleshot(
        drive_vrcaps=Path("/content/drive/MyDrive/vrcaps"),
        use_content_encoder=use_content_encoder,
        controlnet_scale=controlnet_scale,
    )

    verts = align_polyp_to_wall(vertices)
    target = polyp_gaze_target(verts)
    radius = mesh_bounds_radius(verts)
    distance = max(radius * 2.8, 1e-3)
    k = pinhole_intrinsics(control_size, control_size)
    c2w = wall_orbit_camera(0.0, obliquity_deg=elevation_deg, distance=distance, target=target)

    rgb, mask, depth = render_vertex_color(
        verts, faces, colors, k, c2w, control_size, control_size
    )
    if mask.max() == 0:
        raise RuntimeError("Kontrol renderi bos — mesh/kamera ayarini kontrol edin.")

    style_img = Image.open(style_ref).convert("RGB")
    control = build_control_image(
        rgb,
        depth,
        mask,
        detector,
        content_ckpt,
    )
    syn = generate_syn(
        styleshot,
        style_img,
        control,
        prompt,
        seed=seed,
        controlnet_scale=controlnet_scale,
    )

    if syn.shape[0] != out_size or syn.shape[1] != out_size:
        syn = np.array(
            Image.fromarray(syn).resize((out_size, out_size), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )

    meta = {
        "style_encoder": style_ckpt,
        "style_ref": str(style_ref),
        "content_encoder": str(content_ckpt) if content_ckpt else None,
        "seed": seed,
        "prompt": prompt,
        "controlnet_scale": controlnet_scale,
        "control_size": control_size,
    }
    return syn, meta
