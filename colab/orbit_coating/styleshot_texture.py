"""StyleShot UV texture generation for single PLY (encoder via gdown, no Drive upload)."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dataclasses import dataclass

from styleshot_helpers import build_control_image, generate_syn, init_styleshot

STYLE_REF_CACHE = Path("/content/vrcaps_checkpoints/kvasir_style_ref.jpg")
DRIVE_VRCAPS = Path("/content/drive/MyDrive/vrcaps")
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


@dataclass
class TextureSession:
    """Paylasilan StyleShot oturumu — ayni mesh kontrolu ile birden fazla stil ref."""

    styleshot: object
    detector: object
    content_ckpt: str | None
    style_ckpt: str
    control: Image.Image


def resolve_style_ref_path(style_ref: Path | str) -> Path:
    """Verilen stil referans yolunu dogrula."""
    path = Path(style_ref)
    if path.exists() and path.stat().st_size > 1_000:
        return path
    raise FileNotFoundError(f"Stil referansi bulunamadi: {path}")


def resolve_drive_style_ref(stem: str, drive_vrcaps: Path | None = None) -> Path:
    """Drive/vrcaps altinda stil referansi (drive_style_refs modulu)."""
    from drive_style_refs import resolve_drive_style_ref as _resolve

    return _resolve(stem, drive_vrcaps)


def open_texture_session(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    use_content_encoder: bool = False,
    controlnet_scale: float = 0.55,
    control_size: int = 512,
    elevation_deg: float = 18.0,
    drive_vrcaps: Path | None = None,
) -> TextureSession:
    """Mesh kontrol haritasini bir kez uret; birden fazla stil ref icin yeniden kullan."""
    from ply_loader import mesh_bounds_radius
    from turntable_render import align_polyp_to_wall, pinhole_intrinsics, polyp_gaze_target, wall_orbit_camera

    drive = Path(drive_vrcaps or DRIVE_VRCAPS)
    styleshot, detector, content_ckpt, style_ckpt, _pipe = init_styleshot(
        drive_vrcaps=drive,
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

    control = build_control_image(
        rgb,
        depth,
        mask,
        detector,
        content_ckpt,
    )
    return TextureSession(
        styleshot=styleshot,
        detector=detector,
        content_ckpt=str(content_ckpt) if content_ckpt else None,
        style_ckpt=style_ckpt,
        control=control,
    )


def _resize_texture(syn: np.ndarray, out_size: int) -> np.ndarray:
    if syn.shape[0] == out_size and syn.shape[1] == out_size:
        return syn
    return np.array(
        Image.fromarray(syn).resize((out_size, out_size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def generate_uv_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    style_ref: Path | str | None = None,
    session: TextureSession | None = None,
    out_size: int = 2048,
    seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
    controlnet_scale: float = 0.55,
    use_content_encoder: bool = False,
    control_size: int = 512,
    elevation_deg: float = 18.0,
) -> tuple[np.ndarray, dict]:
    """StyleShot ile UV atlas dokusu uret."""
    if session is None:
        session = open_texture_session(
            vertices,
            faces,
            colors,
            use_content_encoder=use_content_encoder,
            controlnet_scale=controlnet_scale,
            control_size=control_size,
            elevation_deg=elevation_deg,
        )

    ref_path = resolve_style_ref_path(style_ref) if style_ref else ensure_style_ref()
    style_img = Image.open(ref_path).convert("RGB")
    syn = generate_syn(
        session.styleshot,
        style_img,
        session.control,
        prompt,
        seed=seed,
        controlnet_scale=controlnet_scale,
    )
    syn_full = _resize_texture(syn, out_size)

    meta = {
        "style_encoder": session.style_ckpt,
        "style_ref": str(ref_path),
        "content_encoder": session.content_ckpt,
        "seed": seed,
        "prompt": prompt,
        "controlnet_scale": controlnet_scale,
        "control_size": control_size,
        "preview_size": int(syn.shape[0]),
    }
    return syn_full, meta


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Gorsel okunamadi: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.array(
        Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def build_style_ref_comparison_strip(
    panels: list[tuple[str, np.ndarray]],
    *,
    panel_size: int = 512,
    gap: int = 12,
    label_height: int = 36,
) -> np.ndarray:
    """Yan yana karsilastirma: ref | uretim | ref | uretim."""
    tiles: list[np.ndarray] = []
    for label, rgb in panels:
        tile = _resize_square(rgb, panel_size)
        banner = np.full((label_height, panel_size, 3), 24, dtype=np.uint8)
        cv2.putText(
            banner,
            label,
            (8, label_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        tiles.append(np.vstack([tile, banner]))

    if not tiles:
        raise ValueError("Karsilastirma paneli bos")

    row_h = tiles[0].shape[0]
    sep = np.full((row_h, gap, 3), 32, dtype=np.uint8)
    out = tiles[0]
    for tile in tiles[1:]:
        out = np.hstack([out, sep, tile])
    return out


def compare_style_refs(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    style_refs: list[tuple[str, Path | str]],
    out_dir: Path,
    *,
    out_size: int = 2048,
    panel_size: int = 512,
    seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
    controlnet_scale: float = 0.55,
    use_content_encoder: bool = False,
    elevation_deg: float = 18.0,
) -> dict:
    """Her stil referansi icin doku uret; 4-panel karsilastirma strip kaydet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = open_texture_session(
        vertices,
        faces,
        colors,
        use_content_encoder=use_content_encoder,
        controlnet_scale=controlnet_scale,
        elevation_deg=elevation_deg,
    )

    panels: list[tuple[str, np.ndarray]] = []
    results: list[dict] = []

    for label, ref in style_refs:
        ref_path = resolve_style_ref_path(ref)
        ref_rgb = _load_rgb(ref_path)
        texture, meta = generate_uv_texture(
            vertices,
            faces,
            colors,
            style_ref=ref_path,
            session=session,
            out_size=out_size,
            seed=seed,
            prompt=prompt,
            controlnet_scale=controlnet_scale,
            use_content_encoder=use_content_encoder,
            elevation_deg=elevation_deg,
        )

        tex_name = f"texture_{label}.png"
        tex_path = out_dir / tex_name
        cv2.imwrite(str(tex_path), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        meta["texture_file"] = tex_name
        results.append(meta)

        panels.append((ref_path.name, ref_rgb))
        panels.append((f"{ref_path.stem} → texture", texture))

    strip = build_style_ref_comparison_strip(panels, panel_size=panel_size)
    strip_path = out_dir / "style_ref_comparison_strip.png"
    cv2.imwrite(str(strip_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    summary = {
        "strip": str(strip_path),
        "panel_size": panel_size,
        "out_size": out_size,
        "seed": seed,
        "prompt": prompt,
        "results": results,
    }
    (out_dir / "style_ref_comparison.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
