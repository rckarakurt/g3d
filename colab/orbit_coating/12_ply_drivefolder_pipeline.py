# 12 — Drive PLY: xatlas UV + StyleShot texture + turntable render
import json
import shutil
import sys
from pathlib import Path

# ============ AYARLAR ============
POLYP_NAME = "polyp_0004.ply"

OUT_ROOT = Path("/content/ply_styleshot_out")
COPY_TO_DRIVE = False  # True = istege bagli Drive yedegi

ATLAS_TEX_SIZE = 2048
TEXTURE_SEED = 42
CONTROLNET_SCALE = 0.55
USE_CONTENT_ENCODER = False

PROMPT = (
    "colonoscopy image of a sessile polyp on pink mucosa, "
    "smooth organic surface, realistic endoscopic texture, clinical photography"
)

AZIMUTH_BIN_DEG = 5          # bank aci adimi (None = otomatik)
AZIMUTH_HALF_SPAN_DEG = 90   # -90 .. +90 (0 = duz yuz, Y ekseni)
SYNC_AZIMUTHS_FROM_UNITY = True
# Oncelik: /content/medical_gan_dataset (Bolum 5b); yoksa Drive fallback
UNITY_DATASET = Path("/content/medical_gan_dataset")

OBLIQUITY_DEG = 18.0  # on gorunumde hafif Y egimi; ±90 yan profilde etkisiz
RENDER_SIZE = 768
WALL_MOUNTED = True
DISTANCE_SCALE = 2.4  # mesh etrafinda donus icin kamera mesafesi
# ================================

import os

os.environ["VRCAPS_USE_KVASIR"] = "0"  # Colab SSL: Kvasir indirme kapali, sentetik ref

os.chdir("/content/StyleShot")
sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

orbit, _ = install_sys_path()
sys.path.insert(0, "/content/StyleShot")

from drive_ply_fetch import fetch_ply_from_drive_folder, load_selected_ply
from ply_loader import center_mesh, load_mesh_ply, mesh_bounds_radius
from mesh_uv import save_uv_preview, unwrap_mesh_uv
from style_encoder_drive import ensure_style_encoder
from turntable_render import wall_azimuth_grid
from unity_view_bank_angles import azimuths_from_gaze_dataset, save_angle_meta
from colab_content_paths import PLY_OUT_DRIVE, copy_tree_if_requested, resolve_unity_dataset
import styleshot_texture as _styleshot_tex


def _patch_style_ref_no_kvasir() -> None:
    """Eski gomulu script olsa bile Kvasir ag indirmesini atla."""
    import cv2
    import numpy as np
    import shutil

    cache = _styleshot_tex.STYLE_REF_CACHE

    def ensure_style_ref(dest=None):
        dest = Path(dest or cache)
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
        for candidate in (
            Path("/content/drive/MyDrive/vrcaps/kvasir_style_ref.jpg"),
            Path("/content/drive/MyDrive/vrcaps/kvasir_style_ref.png"),
        ):
            if candidate.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, dest)
                return dest
        if hasattr(_styleshot_tex, "_synthetic_mucosa_style_ref"):
            return _styleshot_tex._synthetic_mucosa_style_ref(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        base = np.array([188, 108, 118], dtype=np.float32)
        img = base + rng.normal(0, 22, (512, 512, 3))
        yy, xx = np.mgrid[0:512, 0:512]
        vignette = 1.0 - 0.35 * np.sqrt(((xx - 256) / 256) ** 2 + ((yy - 256) / 256) ** 2)
        img = np.clip(img * vignette[..., None], 0, 255).astype(np.uint8)
        cv2.imwrite(str(dest), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("Sentetik stil referansi (Kvasir atlandi):", dest)
        return dest

    _styleshot_tex.ensure_style_ref = ensure_style_ref
    print("Kvasir indirme devre disi — sentetik stil ref kullanilacak")


_patch_style_ref_no_kvasir()
from styleshot_texture import generate_uv_texture
from turntable_render import render_turntable_views

import cv2
import numpy as np

PLY_PATH = load_selected_ply()
if PLY_PATH is None:
    print("Bolum 2 calistirilmamis — PLY simdi indiriliyor...")
    PLY_PATH = fetch_ply_from_drive_folder(POLYP_NAME)

print("Style encoder indiriliyor / kontrol...")
style_encoder_path = ensure_style_encoder()
print("Encoder:", style_encoder_path)

OUT_ROOT.mkdir(parents=True, exist_ok=True)
unity_dataset = resolve_unity_dataset()
uv_dir = OUT_ROOT / "mesh_uv"
tex_dir = OUT_ROOT / "uv_texture"
view_dir = OUT_ROOT / "view_bank"

print("1/4 PLY yukleniyor:", PLY_PATH)
verts, faces, _normals, colors = load_mesh_ply(PLY_PATH)
verts, _center = center_mesh(verts)
radius_mm = mesh_bounds_radius(verts)
print(f"   {len(verts)} vert, {len(faces)} tri, radius ~{radius_mm:.2f} mm")

print("2/4 xatlas UV unwrap...")
verts_uv, faces_uv, uvs = unwrap_mesh_uv(verts, faces, atlas_size=ATLAS_TEX_SIZE)
uv_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    uv_dir / "polyp_uv.npz",
    vertices=verts_uv,
    faces=faces_uv,
    uvs=uvs,
    tex_size=ATLAS_TEX_SIZE,
    source_ply=str(PLY_PATH),
)
blank = np.full((ATLAS_TEX_SIZE, ATLAS_TEX_SIZE, 3), 180, dtype=np.uint8)
save_uv_preview(blank, uv_dir / "uv_layout_preview.png")
print("   ->", uv_dir / "polyp_uv.npz")

print("3/4 StyleShot texture uretiliyor...")
texture, tex_meta = generate_uv_texture(
    verts,
    faces,
    colors,
    out_size=ATLAS_TEX_SIZE,
    seed=TEXTURE_SEED,
    prompt=PROMPT,
    controlnet_scale=CONTROLNET_SCALE,
    use_content_encoder=USE_CONTENT_ENCODER,
    elevation_deg=OBLIQUITY_DEG,
)
tex_dir.mkdir(parents=True, exist_ok=True)
tex_path = tex_dir / "polyp_uv_texture.png"
cv2.imwrite(str(tex_path), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
save_uv_preview(texture, tex_dir / "polyp_uv_texture_preview.png")
(tex_dir / "texture_meta.json").write_text(json.dumps(tex_meta, indent=2), encoding="utf-8")
print("   ->", tex_path)

if SYNC_AZIMUTHS_FROM_UNITY and unity_dataset.exists() and (unity_dataset / "rgb").exists():
    gaze_csv = unity_dataset / "poses" / "gaze_views.csv"
    if not gaze_csv.exists() and (unity_dataset / "rgb").exists():
        from unity_dataset_angles import ensure_geometric_gaze_views

        print("gaze_views.csv yok — Unity dataset uzerinde uretiliyor...")
        ensure_geometric_gaze_views(unity_dataset, write_plot=False)
    AZIMUTHS_DEG, angle_meta = azimuths_from_gaze_dataset(
        unity_dataset,
        bin_deg=AZIMUTH_BIN_DEG,
        half_span_deg=AZIMUTH_HALF_SPAN_DEG,
        fill_span=True,
    )
    save_angle_meta(angle_meta, view_dir / "unity_angle_sync.json")
    print(
        f"Unity aci senkron: {len(AZIMUTHS_DEG)} gorunum, "
        f"bank span {angle_meta['bank_az_min_deg']:.1f}.."
        f"{angle_meta['bank_az_max_deg']:.1f}° (0=duz yuz), bin={angle_meta['bin_deg']}°"
    )
else:
    step = int(AZIMUTH_BIN_DEG or 5)
    AZIMUTHS_DEG = wall_azimuth_grid(step, int(AZIMUTH_HALF_SPAN_DEG))
    print(
        f"Unity dataset yok — sabit grid -{AZIMUTH_HALF_SPAN_DEG}..+{AZIMUTH_HALF_SPAN_DEG} "
        f"step={step}° ({len(AZIMUTHS_DEG)} gorunum)"
    )

print(f"4/4 Duvar-yapisik render ({len(AZIMUTHS_DEG)} aci): ilk/son = {AZIMUTHS_DEG[0]:.0f}..{AZIMUTHS_DEG[-1]:.0f}")
manifest = render_turntable_views(
    verts_uv,
    faces_uv,
    uvs,
    texture,
    AZIMUTHS_DEG,
    view_dir,
    width=RENDER_SIZE,
    height=RENDER_SIZE,
    elevation_deg=OBLIQUITY_DEG,
    distance_scale=DISTANCE_SCALE,
    wall_mounted=WALL_MOUNTED,
    orbit_mode="lumen",  # Y ekseni -90..+90, 0=duz yuz (+Z)
)
print("   ->", view_dir)
for entry in manifest["views"]:
    print(f"      az={entry['azimuth_deg']:5.0f}  {entry['file']}")

print("5/5 Onizleme icin Bolum 4 calistirin (15° grid strip).")

meta = {
    "ply": str(PLY_PATH),
    "style_encoder": str(style_encoder_path),
    "texture_meta": tex_meta,
    "atlas_tex_size": ATLAS_TEX_SIZE,
    "azimuths_deg": AZIMUTHS_DEG,
    "azimuth_bin_deg": AZIMUTH_BIN_DEG,
    "azimuth_half_span_deg": AZIMUTH_HALF_SPAN_DEG,
    "sync_azimuths_from_unity": SYNC_AZIMUTHS_FROM_UNITY,
    "unity_dataset": str(unity_dataset),
    "obliquity_deg": OBLIQUITY_DEG,
    "wall_mounted": WALL_MOUNTED,
    "mesh_radius_mm": radius_mm,
}
(OUT_ROOT / "pipeline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Cikti (content):", OUT_ROOT)
print("  view_bank:", view_dir)
copy_tree_if_requested(OUT_ROOT, PLY_OUT_DRIVE, enabled=COPY_TO_DRIVE)

print("Tamamlandi.")

