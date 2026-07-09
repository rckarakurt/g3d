# 16 — Sabit Unity frame + ayni acili texture polyp (yan yana strip)
from __future__ import annotations

import json
import sys
from pathlib import Path

get_ipython().run_line_magic("pip", "install -q opencv-python-headless")

# ============ AYARLAR ============
# Geometric bank azimuth ile eslesen sabit Unity kareleri + view bank acisi
FIXED_PAIRS: list[dict] = [
    {"frame": 67, "az_deg": -45.0, "image_name": "image_0067.png"},
    {"frame": 78, "az_deg": -30.0, "image_name": "image_0078.png"},
    {"frame": 87, "az_deg": -15.0, "image_name": "image_0087.png"},
    {"frame": 98, "az_deg": 0.0, "image_name": "image_0098.png"},
    {"frame": 121, "az_deg": 15.0, "image_name": "image_0121.png"},
    {"frame": 132, "az_deg": 30.0, "image_name": "image_0132.png"},
    {"frame": 143, "az_deg": 45.0, "image_name": "image_0143.png"},
]

OUT_DIR = Path("/content/unity_polyp_angle_pairs")
CELL_W = 360
CELL_H = 280
LABEL_H = 56
PAD = 10
GAP = 8
BG = (22, 22, 26)
COPY_TO_DRIVE = False
# ================================

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()

import cv2
import numpy as np

from colab_content_paths import (
    DATASET_DRIVE,
    UNITY_DATASET,
    copy_tree_if_requested,
    drive_mounted,
    resolve_unity_dataset,
    resolve_view_bank,
    sync_dataset_from_drive,
)
from turntable_render import azimuth_view_filename
from unity_dataset_angles import ensure_geometric_gaze_views


def ensure_dataset() -> Path:
    if (UNITY_DATASET / "rgb").is_dir():
        return UNITY_DATASET
    if drive_mounted() and (DATASET_DRIVE / "rgb").is_dir():
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        return sync_dataset_from_drive(UNITY_DATASET)
    return resolve_unity_dataset()


def load_rgba_bgra(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        g = img
        return np.dstack([g, g, g, np.full_like(g, 255)])
    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    bgra = img
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    return np.dstack([rgb, bgra[:, :, 3]])


def rgba_on_bg(rgba: np.ndarray, bg: tuple[int, int, int] = BG) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg_arr = np.array(bg, dtype=np.float32)
    out = rgb * a + bg_arr * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def fit_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[2] == 4:
        img = rgba_on_bg(img)
    elif img.shape[2] == 3 and img.dtype == np.uint8:
        pass
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def resolve_polyp_path(view_bank: Path, az_deg: float, manifest: dict) -> Path:
    """Exact bank file for signed azimuth."""
    fname = azimuth_view_filename(az_deg)
    direct = view_bank / fname
    if direct.exists():
        return direct
    for entry in manifest.get("views", []):
        if int(round(float(entry["azimuth_deg"]))) == int(round(float(az_deg))):
            candidate = view_bank / entry["file"]
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"View bank'de {az_deg:+.0f}° yok: {fname}\n"
        f"Once Bolum 3'u -90..+90 arasi 5° adimla calistirin."
    )


def load_unity_rgb(dataset: Path, frame: int, image_name: str) -> np.ndarray:
    candidates = [
        dataset / "rgb" / f"{frame:06d}.png",
        dataset / "rgb" / image_name,
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Unity RGB yok: frame {frame} ({image_name})")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_label(canvas: np.ndarray, x: int, y: int, w: int, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x + 8, y + 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 240),
            1,
            cv2.LINE_AA,
        )


def build_pair_strip(
    dataset: Path,
    view_bank: Path,
    pairs: list[dict],
    *,
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
) -> tuple[np.ndarray, list[dict]]:
    manifest = json.loads((view_bank / "view_manifest.json").read_text(encoding="utf-8"))
    n = len(pairs)
    pair_w = cell_w * 2 + GAP
    title_h = 44
    width = PAD + n * (pair_w + PAD)
    height = title_h + LABEL_H + cell_h + PAD
    canvas = np.full((height, width, 3), BG, dtype=np.uint8)

    cv2.putText(
        canvas,
        "Unity mucosa  |  StyleShot polyp (view bank)  —  fixed frame + angle pairs",
        (PAD, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (200, 200, 210),
        1,
        cv2.LINE_AA,
    )

    meta_rows: list[dict] = []
    y_img = title_h + LABEL_H

    for col, pair in enumerate(pairs):
        frame = int(pair["frame"])
        az = float(pair["az_deg"])
        image_name = str(pair["image_name"])
        x0 = PAD + col * (pair_w + PAD)

        unity_rgb = load_unity_rgb(dataset, frame, image_name)
        polyp_path = resolve_polyp_path(view_bank, az, manifest)
        polyp_rgba = load_rgba_bgra(polyp_path)

        unity_panel = fit_rgb(unity_rgb, cell_w, cell_h)
        polyp_panel = fit_rgb(polyp_rgba, cell_w, cell_h)

        x_unity = x0
        x_polyp = x0 + cell_w + GAP
        canvas[y_img : y_img + cell_h, x_unity : x_unity + cell_w] = unity_panel
        canvas[y_img : y_img + cell_h, x_polyp : x_polyp + cell_w] = polyp_panel

        draw_label(
            canvas,
            x0,
            title_h,
            pair_w,
            [
                f"{az:+.0f} deg",
                f"Unity {image_name}",
                f"Polyp {polyp_path.name}",
            ],
        )

        cv2.putText(
            canvas,
            "Unity",
            (x_unity + 6, y_img + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Polyp",
            (x_polyp + 6, y_img + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (x0, title_h),
            (x0 + pair_w - 1, y_img + cell_h - 1),
            (70, 70, 80),
            1,
        )

        meta_rows.append(
            {
                "frame": frame,
                "image_name": image_name,
                "bank_az_deg": az,
                "polyp_file": polyp_path.name,
                "unity_rgb": str(
                    next(
                        p
                        for p in (
                            dataset / "rgb" / f"{frame:06d}.png",
                            dataset / "rgb" / image_name,
                        )
                        if p.exists()
                    )
                ),
            }
        )
        print(
            f"  {az:+5.0f}°  Unity {image_name}  |  {polyp_path.name}"
        )

    return canvas, meta_rows


def show_preview(path: Path) -> None:
    from IPython.display import Image, display

    print(path.name)
    display(Image(filename=str(path), width=1200))


dataset = ensure_dataset()
view_bank = resolve_view_bank()
if not (view_bank / "view_manifest.json").exists():
    raise FileNotFoundError(f"View bank yok: {view_bank}\nOnce Bolum 3 calistirin.")

print("Dataset:", dataset)
print("View bank:", view_bank)
print("\nGaze acilari (dogrulama)...")
ensure_geometric_gaze_views(dataset, write_plot=False, force=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("\nSabit ciftler:")
strip_rgb, pair_meta = build_pair_strip(dataset, view_bank, FIXED_PAIRS)

strip_path = OUT_DIR / "angle_pairs_strip.png"
cv2.imwrite(str(strip_path), cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2BGR))
(OUT_DIR / "pairs_manifest.json").write_text(
    json.dumps({"pairs": pair_meta}, indent=2),
    encoding="utf-8",
)

for item in pair_meta:
    frame = int(item["frame"])
    az = float(item["bank_az_deg"])
    unity = load_unity_rgb(dataset, frame, item["image_name"])
    polyp = rgba_on_bg(load_rgba_bgra(view_bank / item["polyp_file"]))
    pair_w = CELL_W * 2 + GAP
    row_h = CELL_H
    pair_img = np.full((row_h, pair_w, 3), BG, dtype=np.uint8)
    pair_img[:, :CELL_W] = fit_rgb(unity, CELL_W, CELL_H)
    pair_img[:, CELL_W + GAP :] = fit_rgb(polyp, CELL_W, CELL_H)
    out_one = OUT_DIR / f"pair_{az:+.0f}deg_f{frame:04d}.png".replace("+", "p").replace("-", "m")
    cv2.imwrite(str(out_one), cv2.cvtColor(pair_img, cv2.COLOR_RGB2BGR))

copy_tree_if_requested(
    OUT_DIR,
    Path("/content/drive/MyDrive/vrcaps/unity_polyp_angle_pairs"),
    enabled=COPY_TO_DRIVE,
)

print("\n=== Tamamlandi ===")
print("Strip:", strip_path)
print("Klasor:", OUT_DIR)
show_preview(strip_path)
