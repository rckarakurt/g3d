# 12b — Iki stil referansi ile texture karsilastirmasi (4-panel strip)
import sys
from pathlib import Path

# ============ AYARLAR ============
POLYP_NAME = "polyp_0004.ply"
OUT_ROOT = Path("/content/ply_styleshot_out")
COMPARE_DIR = OUT_ROOT / "style_ref_compare"

ATLAS_TEX_SIZE = 2048
PANEL_SIZE = 512
TEXTURE_SEED = 42
CONTROLNET_SCALE = 0.55
USE_CONTENT_ENCODER = False
OBLIQUITY_DEG = 18.0

PROMPT = (
    "colonoscopy image of a sessile polyp on pink mucosa, "
    "smooth organic surface, realistic endoscopic texture, clinical photography"
)

DRIVE_VRCAPS = Path("/content/drive/MyDrive/vrcaps")  # resolve_drive_vrcaps_root() ile guncellenir
STYLE_REF_STEMS = ("kvasir_style_ref", "polyp_texture")
# ================================

import os

os.environ["VRCAPS_USE_KVASIR"] = "0"

os.chdir("/content/StyleShot")
sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path, resolve_drive_vrcaps_root

install_sys_path()
sys.path.insert(0, "/content/StyleShot")

from drive_ply_fetch import fetch_ply_from_drive_folder, load_selected_ply
from ply_loader import center_mesh, load_mesh_ply
from style_encoder_drive import ensure_style_encoder
from styleshot_texture import compare_style_refs, resolve_drive_style_ref

PLY_PATH = load_selected_ply()
if PLY_PATH is None:
    print("Bolum 2 calistirilmamis — PLY simdi indiriliyor...")
    PLY_PATH = fetch_ply_from_drive_folder(POLYP_NAME)

print("Style encoder kontrol...")
print("Encoder:", ensure_style_encoder())

DRIVE_VRCAPS = resolve_drive_vrcaps_root(mount=True)
print("Drive vrcaps:", DRIVE_VRCAPS)
try:
    ref_listing = sorted(p.name for p in DRIVE_VRCAPS.iterdir() if p.is_file())
    print("vrcaps gorseller:", ", ".join(ref_listing[:20]))
except OSError:
    pass

style_refs: list[tuple[str, Path]] = []
for stem in STYLE_REF_STEMS:
    ref_path = resolve_drive_style_ref(stem, DRIVE_VRCAPS)
    style_refs.append((stem, ref_path))
    print("Stil ref:", ref_path)

print("PLY yukleniyor:", PLY_PATH)
verts, faces, _normals, colors = load_mesh_ply(PLY_PATH)
verts, _center = center_mesh(verts)

print("StyleShot — her referans icin texture + 4-panel strip...")
summary = compare_style_refs(
    verts,
    faces,
    colors,
    style_refs,
    COMPARE_DIR,
    out_size=ATLAS_TEX_SIZE,
    panel_size=PANEL_SIZE,
    seed=TEXTURE_SEED,
    prompt=PROMPT,
    controlnet_scale=CONTROLNET_SCALE,
    use_content_encoder=USE_CONTENT_ENCODER,
    elevation_deg=OBLIQUITY_DEG,
)

strip_path = Path(summary["strip"])
print("\nKarsilastirma strip:", strip_path)
for entry in summary["results"]:
    print(f"  {entry['style_ref']} -> {COMPARE_DIR / entry['texture_file']}")

try:
    from IPython.display import Image as IPImage, display

    display(IPImage(filename=str(strip_path)))
except Exception:
    pass

print("\nTamamlandi:", COMPARE_DIR)
