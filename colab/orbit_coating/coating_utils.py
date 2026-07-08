"""Poisson blend, LAB color match, MP4 export, Stage-2 encoder helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def resolve_content_encoder(drive_vrcaps: Path) -> Path | None:
    for p in (
        drive_vrcaps / 'styleshot_polyp_finetuned/content_fusion_encoder_polyp.bin',
        drive_vrcaps / 'content_fusion_encoder_polyp.bin',
        Path('checkpoints/polyp_finetuned/content_fusion_encoder_polyp.bin'),
    ):
        if p.exists() and p.stat().st_size > 100_000:
            return p
    return None


def build_hed_content(rgb: np.ndarray, detector) -> "Image.Image":
    from PIL import Image

    hed = detector(rgb)
    return Image.fromarray(hed)


def align_syn_to_mask(syn_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    th, tw = mask.shape[:2]
    h, w = syn_rgb.shape[:2]
    if (h, w) != (th, tw):
        syn_rgb = cv2.resize(syn_rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    return syn_rgb


def green_pixels_mask(rgb: np.ndarray) -> np.ndarray:
    """Unity segmentasyon yesili — kaplama temizligi icin genis HSV araligi."""
    if cv2 is None:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, (25, 35, 35), (98, 255, 255))


def expand_coating_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    dilate_iters: int = 8,
    catch_green: bool = True,
) -> np.ndarray:
    """Polyp maskesini genislet; komsu yesil pikselleri de kaplama alanina al."""
    if cv2 is None or mask.max() == 0:
        return mask
    coat = mask.copy()
    if dilate_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        coat = cv2.dilate(coat, k, iterations=dilate_iters)
    if catch_green:
        green = green_pixels_mask(rgb)
        near = cv2.dilate(coat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 1)
        extra = (green > 0) & (near > 0)
        coat = np.where(extra, 255, coat).astype(np.uint8)
    return coat


def prepare_syn_canvas(
    unity_rgb: np.ndarray,
    syn_rgb: np.ndarray,
    coat_mask: np.ndarray,
) -> np.ndarray:
    """Sentetik doku yalnizca kaplama maskesi icinde; disari Unity."""
    syn_rgb = align_syn_to_mask(syn_rgb, coat_mask)
    canvas = unity_rgb.copy()
    poly = coat_mask > 0
    canvas[poly] = syn_rgb[poly]
    return canvas


def strip_green_from_syn(syn_rgb: np.ndarray, coat_mask: np.ndarray) -> np.ndarray:
    """StyleShot ciktisindaki yesil spill'i polyp icinde temizle."""
    if cv2 is None or coat_mask.max() == 0:
        return syn_rgb
    out = syn_rgb.copy()
    green = (green_pixels_mask(syn_rgb) > 0) & (coat_mask > 0)
    if not green.any():
        return out
    poly = coat_mask > 0
    ref = out[poly & ~green]
    if ref.size == 0:
        ref = out[poly]
    fill = ref.mean(axis=0).astype(np.uint8)
    out[green] = fill
    out = cv2.inpaint(
        out,
        green.astype(np.uint8) * 255,
        5,
        cv2.INPAINT_TELEA,
    )
    return out


def suppress_residual_green(
    rgb: np.ndarray,
    syn_rgb: np.ndarray,
    coat_mask: np.ndarray,
) -> np.ndarray:
    """Kompozit sonrasi kalan yesil pikselleri sentetik doku ile boya."""
    if cv2 is None or coat_mask.max() == 0:
        return rgb
    green = (green_pixels_mask(rgb) > 0) & (coat_mask > 0)
    if not green.any():
        return rgb
    out = rgb.copy()
    syn_rgb = align_syn_to_mask(syn_rgb, coat_mask)
    out[green] = syn_rgb[green]
    return out


def make_coating_alpha(
    coat_mask: np.ndarray,
    *,
    core_erode: int = 5,
    feather: int = 21,
) -> np.ndarray:
    """Ic bolge tam sentetik (alpha=1); yalnizca dis sinirda yumusak gecis."""
    if cv2 is None:
        m = (coat_mask > 0).astype(np.float32)
        return m[..., None]
    m = coat_mask.astype(np.float32) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core = cv2.erode(coat_mask, k, iterations=max(core_erode, 1))
    core_f = core.astype(np.float32) / 255.0
    rim = np.clip(m - core_f, 0.0, 1.0)
    f = feather if feather % 2 == 1 else feather + 1
    rim_soft = cv2.GaussianBlur(rim, (f, f), 0)
    alpha = np.clip(core_f + rim_soft, 0.0, 1.0)
    return alpha[..., None]


def lab_color_match_polyp(
    syn_rgb: np.ndarray,
    unity_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.55,
) -> np.ndarray:
    """Match syn polyp LAB statistics to Unity mucosa rim (just outside polyp)."""
    if cv2 is None or strength <= 0 or mask.max() == 0:
        return syn_rgb
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(mask, k, iterations=2)
    rim = (dilated > 0) & (mask == 0)
    green_rim = (green_pixels_mask(unity_rgb) > 0) & (dilated > 0)
    rim = rim & ~green_rim
    poly = mask > 0
    if not rim.any() or not poly.any():
        return syn_rgb
    syn_lab = cv2.cvtColor(syn_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(unity_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean = syn_lab[poly].mean(axis=0)
    ref_mean = ref_lab[rim].mean(axis=0)
    delta = (ref_mean - src_mean) * strength
    out_lab = syn_lab.copy()
    out_lab[poly] = np.clip(out_lab[poly] + delta, 0, 255)
    return cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def gaussian_blend_polyp(
    unity_rgb: np.ndarray,
    syn_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    mask_dilate: int = 5,
    feather: int = 31,
) -> np.ndarray:
    coat = expand_coating_mask(unity_rgb, mask, dilate_iters=mask_dilate)
    syn_rgb = strip_green_from_syn(align_syn_to_mask(syn_rgb, coat), coat)
    canvas = prepare_syn_canvas(unity_rgb, syn_rgb, coat)
    alpha = make_coating_alpha(coat, core_erode=4, feather=feather)
    out = np.clip(
        unity_rgb.astype(np.float32) * (1.0 - alpha) + canvas.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    return suppress_residual_green(out, syn_rgb, coat)


def poisson_rim_blend(
    base_rgb: np.ndarray,
    syn_rgb: np.ndarray,
    coat_mask: np.ndarray,
    *,
    rim_erode: int = 10,
) -> np.ndarray:
    """Yalnizca dar sinir bandinda NORMAL_CLONE — yesil gradyan karismaz."""
    if cv2 is None or coat_mask.max() == 0:
        return base_rgb
    h, w = base_rgb.shape[:2]
    syn_rgb = align_syn_to_mask(syn_rgb, coat_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner = cv2.erode(coat_mask, k, iterations=max(rim_erode, 3))
    rim = cv2.subtract(coat_mask, inner)
    rim = cv2.GaussianBlur(rim, (11, 11), 0)
    rim_u8 = np.where(rim > 48, 255, 0).astype(np.uint8)
    if rim_u8.max() == 0:
        return base_rgb
    moments = cv2.moments(rim_u8)
    if moments['m00'] < 1:
        return base_rgb
    cx = int(np.clip(moments['m10'] / moments['m00'], 1, w - 2))
    cy = int(np.clip(moments['m01'] / moments['m00'], 1, h - 2))
    try:
        out_bgr = cv2.seamlessClone(
            cv2.cvtColor(syn_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR),
            rim_u8,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return base_rgb


def poisson_blend_polyp(
    unity_rgb: np.ndarray,
    syn_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    mask_dilate: int = 8,
    clone_mode: str = 'mixed',
) -> np.ndarray:
    """Legacy wrapper — yesil uzerine tam kaplama."""
    del clone_mode
    return blend_polyp(
        unity_rgb,
        syn_rgb,
        mask,
        use_poisson=True,
        mask_dilate=max(mask_dilate, 6),
        feather=21,
    )


def blend_polyp(
    unity_rgb: np.ndarray,
    syn_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    use_poisson: bool = True,
    lab_match: bool = True,
    lab_strength: float = 0.55,
    mask_dilate: int = 8,
    feather: int = 21,
) -> np.ndarray:
    """Yesil Unity polyp bolgesini trajectory-sentetik doku ile tamamen degistir."""
    if cv2 is None:
        return syn_rgb
    coat = expand_coating_mask(unity_rgb, mask, dilate_iters=mask_dilate, catch_green=True)
    syn_rgb = align_syn_to_mask(syn_rgb, coat)
    if lab_match:
        syn_rgb = lab_color_match_polyp(syn_rgb, unity_rgb, coat, strength=lab_strength)
    syn_rgb = strip_green_from_syn(syn_rgb, coat)
    canvas = prepare_syn_canvas(unity_rgb, syn_rgb, coat)
    alpha = make_coating_alpha(coat, core_erode=5, feather=feather)
    out = np.clip(
        unity_rgb.astype(np.float32) * (1.0 - alpha) + canvas.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    out = suppress_residual_green(out, syn_rgb, coat)
    if use_poisson:
        out = poisson_rim_blend(out, syn_rgb, coat, rim_erode=9)
        out = suppress_residual_green(out, syn_rgb, coat)
    return out


def export_trajectory_mp4(
    out_dir: Path,
    *,
    pattern: str = '*_composite.png',
    fps: float = 10.0,
    output_name: str = 'trajectory_coating.mp4',
) -> Path | None:
    if cv2 is None:
        return None
    frames = sorted(out_dir.glob(pattern))
    if len(frames) < 2:
        return None
    first = cv2.imread(str(frames[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    mp4_path = out_dir / output_name
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h),
    )
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(img)
    writer.release()
    return mp4_path if mp4_path.exists() else None
