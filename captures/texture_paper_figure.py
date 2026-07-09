"""Paper figure: StyleShot style reference vs generated UV texture + validation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BG = (248, 248, 250)
HEADER = (32, 32, 36)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def _header_bar(width: int, title: str, *, height: int = 52) -> np.ndarray:
    bar = np.full((height, width, 3), HEADER, dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (12, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return bar


def build_texture_paper_table(
    style_ref_rgb: np.ndarray,
    texture_rgb: np.ndarray,
    *,
    panel_size: int = 640,
    gap: int = 16,
    left_title: str = "Style reference",
    right_title: str = "StyleShot UV texture",
) -> np.ndarray:
    """Two-column publication table: reference | generated texture."""
    left = _resize_square(style_ref_rgb, panel_size)
    right = _resize_square(texture_rgb, panel_size)
    left_col = np.vstack([_header_bar(panel_size, left_title), left])
    right_col = np.vstack([_header_bar(panel_size, right_title), right])
    sep = np.full((left_col.shape[0], gap, 3), BG, dtype=np.uint8)
    row = np.hstack([left_col, sep, right_col])
    pad = np.full((gap, row.shape[1], 3), BG, dtype=np.uint8)
    return np.vstack([pad, row, pad])


def _lab_stats(rgb: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return {
        "L_mean": float(mean[0]),
        "a_mean": float(mean[1]),
        "b_mean": float(mean[2]),
        "L_std": float(std[0]),
        "a_std": float(std[1]),
        "b_std": float(std[2]),
    }


def _histogram_correlation(a_gray: np.ndarray, b_gray: np.ndarray, bins: int = 64) -> float:
    ha = cv2.calcHist([a_gray], [0], None, [bins], [0, 256]).astype(np.float64).flatten()
    hb = cv2.calcHist([b_gray], [0], None, [bins], [0, 256]).astype(np.float64).flatten()
    ha /= max(ha.sum(), 1e-6)
    hb /= max(hb.sum(), 1e-6)
    return float(np.corrcoef(ha, hb)[0, 1])


def _green_fraction(rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (25, 35, 35), (98, 255, 255))
    return float((mask > 0).mean())


def _shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).astype(np.float64).flatten()
    p = hist / max(hist.sum(), 1e-6)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    return float(ssim(a.astype(np.float64), b.astype(np.float64), data_range=255.0))


def validate_uv_texture(
    style_ref_rgb: np.ndarray,
    texture_rgb: np.ndarray,
    *,
    view_bank_rgb: np.ndarray | None = None,
) -> dict:
    """Lightweight QA metrics for StyleShot UV texture (paper / ablation)."""
    ref_sq = _resize_square(style_ref_rgb, 512)
    tex_sq = _resize_square(texture_rgb, 512)
    ref_gray = cv2.cvtColor(ref_sq, cv2.COLOR_RGB2GRAY)
    tex_gray = cv2.cvtColor(tex_sq, cv2.COLOR_RGB2GRAY)

    ref_stats = _lab_stats(ref_sq)
    tex_stats = _lab_stats(tex_sq)
    lab_delta = {
        k: float(tex_stats[k] - ref_stats[k])
        for k in ("L_mean", "a_mean", "b_mean")
    }
    lab_delta["deltaE_approx"] = float(
        np.sqrt(lab_delta["L_mean"] ** 2 + lab_delta["a_mean"] ** 2 + lab_delta["b_mean"] ** 2)
    )

    metrics: dict = {
        "ref_lab": ref_stats,
        "texture_lab": tex_stats,
        "lab_delta": lab_delta,
        "luminance_hist_corr": _histogram_correlation(ref_gray, tex_gray),
        "texture_entropy_bits": _shannon_entropy(tex_gray),
        "texture_green_fraction": _green_fraction(tex_sq),
        "texture_resolution": [int(texture_rgb.shape[1]), int(texture_rgb.shape[0])],
    }

    ssim_ref_tex = _ssim_gray(ref_gray, tex_gray)
    if ssim_ref_tex is not None:
        metrics["ssim_ref_vs_texture"] = ssim_ref_tex

    if view_bank_rgb is not None:
        view_sq = _resize_square(view_bank_rgb, 512)
        view_gray = cv2.cvtColor(view_sq, cv2.COLOR_RGB2GRAY)
        metrics["luminance_hist_corr_texture_vs_view0"] = _histogram_correlation(
            tex_gray, view_gray
        )
        ssim_view = _ssim_gray(tex_gray, view_gray)
        if ssim_view is not None:
            metrics["ssim_texture_vs_view0"] = ssim_view

    metrics["interpretation"] = {
        "luminance_hist_corr": "1.0 = same brightness distribution; 0.3-0.7 typical after style transfer",
        "lab_delta": "Lower deltaE_approx = closer mucosa colour to reference",
        "texture_green_fraction": "Should be near 0 (segmentation green spill)",
        "texture_entropy_bits": "Higher = richer micro-texture (very low may indicate flat failure)",
        "ssim_ref_vs_texture": "Not expected to be high (different geometry); use view-bank SSIM instead",
    }
    return metrics


def resolve_style_ref_from_meta(tex_dir: Path, meta: dict) -> Path:
    style_ref = meta.get("style_ref")
    if style_ref:
        path = Path(style_ref)
        if path.is_file():
            return path
    for candidate in (
        Path("/content/vrcaps_checkpoints/kvasir_style_ref.jpg"),
        Path("/content/drive/MyDrive/vrcaps/kvasir_style_ref.jpg"),
        Path("/content/drive/MyDrive/vrcaps/kvasir_style_ref.png"),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Style reference not found. texture_meta.json style_ref={style_ref!r}"
    )


def export_texture_paper_figure(
    tex_dir: Path,
    out_dir: Path,
    *,
    panel_size: int = 640,
    view_bank_dir: Path | None = None,
) -> dict:
    """Build paper table + validation report from Bolum 3 uv_texture output."""
    tex_dir = Path(tex_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    texture_path = tex_dir / "polyp_uv_texture.png"
    meta_path = tex_dir / "texture_meta.json"
    if not texture_path.is_file():
        raise FileNotFoundError(f"UV texture missing: {texture_path}\nRun Bolum 3 first.")
    if not meta_path.is_file():
        raise FileNotFoundError(f"texture_meta.json missing: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    style_ref_path = resolve_style_ref_from_meta(tex_dir, meta)
    style_ref_rgb = _load_rgb(style_ref_path)
    texture_rgb = _load_rgb(texture_path)

    view_rgb = None
    view_file = None
    if view_bank_dir is not None:
        view_bank_dir = Path(view_bank_dir)
        manifest_path = view_bank_dir / "view_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest.get("views", []):
                if abs(float(entry.get("azimuth_deg", 999))) < 0.6:
                    view_file = view_bank_dir / entry["file"]
                    break
            if view_file is None and manifest.get("views"):
                mid = min(
                    manifest["views"],
                    key=lambda v: abs(float(v.get("azimuth_deg", 0))),
                )
                view_file = view_bank_dir / mid["file"]
        if view_file is not None and view_file.is_file():
            view_bgra = cv2.imread(str(view_file), cv2.IMREAD_UNCHANGED)
            if view_bgra is not None:
                if view_bgra.shape[2] == 4:
                    alpha = view_bgra[:, :, 3:4].astype(np.float32) / 255.0
                    rgb = view_bgra[:, :, :3].astype(np.float32)
                    bg = np.full_like(rgb, 180.0)
                    view_rgb = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
                    view_rgb = cv2.cvtColor(view_rgb, cv2.COLOR_BGR2RGB)
                else:
                    view_rgb = cv2.cvtColor(view_bgra, cv2.COLOR_BGR2RGB)

    table = build_texture_paper_table(style_ref_rgb, texture_rgb, panel_size=panel_size)
    table_path = out_dir / "paper_texture_ref_table.png"
    cv2.imwrite(str(table_path), cv2.cvtColor(table, cv2.COLOR_RGB2BGR))

    metrics = validate_uv_texture(style_ref_rgb, texture_rgb, view_bank_rgb=view_rgb)
    summary = {
        "style_ref": str(style_ref_path),
        "texture": str(texture_path),
        "table_figure": str(table_path),
        "panel_size": panel_size,
        "texture_meta": meta,
        "view_bank_file": str(view_file) if view_file else None,
        "validation": metrics,
        "validation_notes": [
            "Qualitative: mucosa tone, vascular detail, absence of green/flat regions.",
            "Quantitative: lab_delta, luminance_hist_corr, texture_entropy, green_fraction.",
            "Geometric: compare view_bank 0 deg render with texture (ssim_texture_vs_view0).",
            "Clinical plausibility: expert rating on 1-5 scale (recommended for paper).",
        ],
    }
    (out_dir / "texture_validation_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
