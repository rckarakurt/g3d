"""Shared StyleShot init + generate helpers for mesh UV bake (Bölüm 8)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, UNet2DConditionModel

from coating_utils import resolve_content_encoder
from medical_utils import disable_nsfw_filter


def resolve_style_encoder_path(
    drive_vrcaps: Path | None = None,
    style_encoder: str | Path | None = None,
) -> str:
    if style_encoder is not None:
        p = Path(style_encoder)
        if p.exists() and p.stat().st_size > 1_000_000:
            return str(p)

    from style_encoder_drive import ensure_style_encoder

    local = ensure_style_encoder()
    if local.exists() and local.stat().st_size > 1_000_000:
        return str(local)

    if drive_vrcaps is not None:
        for p in (
            drive_vrcaps / "styleshot_polyp_finetuned/style_aware_encoder_polyp.bin",
            drive_vrcaps / "styleshot_polyp_finetuned/style_encoder_final.bin",
            drive_vrcaps / "style_encoder_final.bin",
        ):
            if p.exists() and p.stat().st_size > 1_000_000:
                return str(p)

    raise FileNotFoundError("Style encoder bulunamadi.")


def init_styleshot(
    drive_vrcaps: Path | None = None,
    *,
    style_encoder: str | Path | None = None,
    use_content_encoder: bool = True,
    controlnet_scale: float = 0.55,
):
    """Return (styleshot, detector, content_ckpt, style_ckpt, pipe)."""
    from annotator.hed import SOFT_HEDdetector
    from ip_adapter import StyleContentStableDiffusionControlNetPipeline, StyleShot

    from styleshot_models import ensure_styleshot_models, ip_bin_path

    ensure_styleshot_models()
    style_ckpt = resolve_style_encoder_path(drive_vrcaps, style_encoder)
    content_ckpt = (
        resolve_content_encoder(drive_vrcaps) if use_content_encoder and drive_vrcaps else None
    )

    sd15 = "runwayml/stable-diffusion-v1-5"
    ip_bin = str(ip_bin_path())
    clip_enc = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    detector = SOFT_HEDdetector()
    unet = UNet2DConditionModel.from_pretrained(
        sd15, subfolder="unet", torch_dtype=torch.float16
    )
    content_enc = ControlNetModel.from_unet(unet)
    if content_ckpt is not None:
        content_enc.load_state_dict(torch.load(content_ckpt, map_location="cpu"))
    pipe = disable_nsfw_filter(
        StyleContentStableDiffusionControlNetPipeline.from_pretrained(
            sd15, controlnet=content_enc, torch_dtype=torch.float16
        ).to("cuda")
    )
    styleshot = StyleShot("cuda", pipe, ip_bin, style_ckpt, clip_enc)
    return styleshot, detector, content_ckpt, style_ckpt, pipe


def build_control_image(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    detector,
    content_ckpt: Path | None,
    *,
    control_hed_bg: float = 0.22,
    control_bg_level: int = 127,
    polyp_rim_weight: int = 35,
    polyp_rim_erode: int = 2,
) -> Image.Image:
    """Build StyleShot control image.

    Stage 2 (content_ckpt): raw HED — matches polyp-finetuned ControlNet training.
    Stage 1: depth + optional HED background + polyp rim emphasis.
    """
    if content_ckpt is not None:
        return Image.fromarray(detector(rgb))

    h, w = rgb.shape[:2]
    polyp = mask > 0
    d_norm = np.zeros((h, w), dtype=np.float32)
    valid = depth > 1e-4
    if valid.any():
        lo, hi = np.percentile(depth[valid], [5, 95])
        d_norm[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_soft = cv2.GaussianBlur(
        (d_norm * 255).astype(np.uint8), (31, 31), 0
    ).astype(np.float32)
    combined = np.full((h, w), control_bg_level, dtype=np.float32)
    combined[polyp] = depth_soft[polyp]
    if control_hed_bg > 0:
        hed = detector(rgb).astype(np.float32)
        bg = ~polyp
        combined[bg] = np.clip(
            combined[bg] * (1 - control_hed_bg) + hed[bg] * control_hed_bg, 0, 255
        )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, k, iterations=polyp_rim_erode)
    rim = (mask > 0) & (eroded == 0)
    if rim.any() and polyp_rim_weight > 0:
        combined[rim] = np.clip(combined[rim] + polyp_rim_weight, 0, 255)
    if polyp.any():
        smooth = cv2.GaussianBlur(combined.astype(np.uint8), (17, 17), 0).astype(
            np.float32
        )
        combined[polyp] = smooth[polyp]
    return Image.fromarray(np.clip(combined, 0, 255).astype(np.uint8))


def generate_syn(
    styleshot,
    style_img: Image.Image,
    control: Image.Image,
    prompt: str,
    *,
    seed: int = 42,
    controlnet_scale: float = 0.55,
) -> np.ndarray:
    kw = dict(
        style_image=style_img,
        prompt=[[prompt]],
        content_image=control,
        controlnet_conditioning_scale=controlnet_scale,
    )
    for extra in ({"seed": seed}, {"random_seed": seed}):
        try:
            out = styleshot.generate(**kw, **extra)
            return np.array(out[0][0].convert("RGB"))
        except TypeError:
            pass
    gen = torch.Generator(device="cuda").manual_seed(seed)
    try:
        out = styleshot.generate(**kw, generator=gen)
    except TypeError:
        out = styleshot.generate(**kw)
    return np.array(out[0][0].convert("RGB"))
