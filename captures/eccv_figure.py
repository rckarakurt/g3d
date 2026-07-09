"""ECCV / Springer LNCS figure layout helpers (300 DPI, 122 mm text width)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# Springer LNCS single-column text width (ECCV main track).
ECCV_TEXTWIDTH_MM = 122.0
ECCV_DPI = 300
ECCV_GAP_MM = 1.2
ECCV_LABEL_PT = 9
ECCV_SUBLABEL_PT = 8
ECCV_BG = (255, 255, 255)
ECCV_RULE = (200, 200, 200)


def mm_to_px(mm: float, *, dpi: int = ECCV_DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def pt_to_px(pt: float, *, dpi: int = ECCV_DPI) -> int:
    return int(round(pt / 72.0 * dpi))


def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _font_scale_for_pt(pt: float, *, dpi: int = ECCV_DPI) -> float:
    return max(pt / 72.0 * dpi / 22.0, 0.35)


def draw_panel_label(
    rgb: np.ndarray,
    label: str,
    *,
    pt: float = ECCV_LABEL_PT,
    dpi: int = ECCV_DPI,
    margin_px: int | None = None,
) -> np.ndarray:
    """ECCV-style subfigure tag, e.g. (a), bottom-left."""
    out = rgb.copy()
    h, w = out.shape[:2]
    margin = margin_px if margin_px is not None else max(4, pt_to_px(2, dpi=dpi))
    scale = _font_scale_for_pt(pt, dpi=dpi)
    thickness = max(1, int(round(scale * 1.6)))
    x = margin
    y = h - margin
    # Thin dark outline for readability on bright tissue.
    cv2.putText(
        out,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def draw_subtitle_bar(
    rgb: np.ndarray,
    text: str,
    *,
    pt: float = ECCV_SUBLABEL_PT,
    dpi: int = ECCV_DPI,
    bar_h_px: int | None = None,
) -> np.ndarray:
    """Small caption strip under a panel (optional; prefer LaTeX \\caption when possible)."""
    h = bar_h_px if bar_h_px is not None else pt_to_px(pt + 4, dpi=dpi)
    w = rgb.shape[1]
    bar = np.full((h, w, 3), ECCV_BG, dtype=np.uint8)
    scale = _font_scale_for_pt(pt, dpi=dpi)
    cv2.putText(
        bar,
        text,
        (4, h - max(4, pt_to_px(2, dpi=dpi))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([rgb, bar])


def hstack_panels(
    panels: Sequence[np.ndarray],
    *,
    gap_mm: float = ECCV_GAP_MM,
    dpi: int = ECCV_DPI,
    rule: bool = True,
) -> np.ndarray:
    if not panels:
        raise ValueError("No panels")
    gap_px = mm_to_px(gap_mm, dpi=dpi)
    out = panels[0]
    for panel in panels[1:]:
        sep = np.full((out.shape[0], gap_px, 3), ECCV_BG, dtype=np.uint8)
        if rule and gap_px >= 2:
            mid = gap_px // 2
            sep[:, mid : mid + 1] = ECCV_RULE
        out = np.hstack([out, sep, panel])
    return out


def full_width_panel_width_mm(n_panels: int, *, gap_mm: float = ECCV_GAP_MM) -> float:
    if n_panels < 1:
        raise ValueError("n_panels must be >= 1")
    total_gap = gap_mm * max(0, n_panels - 1)
    return (ECCV_TEXTWIDTH_MM - total_gap) / n_panels


def build_full_width_row(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    aspect: float = 1.0,
    subtitles: Sequence[str] | None = None,
    gap_mm: float = ECCV_GAP_MM,
    dpi: int = ECCV_DPI,
) -> np.ndarray:
    """Assemble ECCV row: equal-width panels, optional subtitles, (a) labels on image."""
    n = len(images)
    if len(labels) != n:
        raise ValueError("labels length must match images")
    if subtitles is not None and len(subtitles) != n:
        raise ValueError("subtitles length must match images")

    panel_w_mm = full_width_panel_width_mm(n, gap_mm=gap_mm)
    panel_w = mm_to_px(panel_w_mm, dpi=dpi)
    panel_h = int(round(panel_w * aspect))

    panels: list[np.ndarray] = []
    for i, img in enumerate(images):
        tile = resize_rgb(img, panel_w, panel_h)
        tile = draw_panel_label(tile, labels[i], dpi=dpi)
        if subtitles is not None and subtitles[i]:
            tile = draw_subtitle_bar(tile, subtitles[i], dpi=dpi)
        panels.append(tile)

    row = hstack_panels(panels, gap_mm=gap_mm, dpi=dpi)
    pad_y = mm_to_px(1.0, dpi=dpi)
    pad = np.full((pad_y, row.shape[1], 3), ECCV_BG, dtype=np.uint8)
    return np.vstack([pad, row, pad])


def save_eccv_figure(
    rgb: np.ndarray,
    path: Path,
    *,
    dpi: int = ECCV_DPI,
    also_pdf: bool = True,
) -> dict[str, str]:
    """Save PNG at 300 DPI; optional PDF vector wrapper via matplotlib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
    try:
        from PIL import Image

        Image.fromarray(rgb).save(png_path, dpi=(dpi, dpi))
    except ImportError:
        cv2.imwrite(str(png_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    outputs = {"png": str(png_path)}
    if also_pdf:
        pdf_path = png_path.with_suffix(".pdf")
        try:
            import matplotlib.pyplot as plt

            h, w = rgb.shape[:2]
            fig_w = w / dpi
            fig_h = h / dpi
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.imshow(rgb)
            ax.set_axis_off()
            fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            outputs["pdf"] = str(pdf_path)
        except Exception:
            pass
    return outputs


def validation_metrics_latex(metrics: dict, *, caption: str = "UV texture validation.") -> str:
    """Booktabs-style LaTeX table for validation metrics (ECCV table environment)."""
    lab = metrics.get("lab_delta", {})
    rows = [
        ("Luminance histogram corr.", f"{metrics.get('luminance_hist_corr', float('nan')):.3f}"),
        (r"Colour $\Delta E$ (approx.)", f"{lab.get('deltaE_approx', float('nan')):.2f}"),
        ("Texture entropy (bits)", f"{metrics.get('texture_entropy_bits', float('nan')):.2f}"),
        ("Green fraction", f"{metrics.get('texture_green_fraction', float('nan')):.4f}"),
    ]
    if "ssim_texture_vs_view0" in metrics:
        rows.append(
            ("SSIM vs.\\ view $0^\\circ$", f"{metrics['ssim_texture_vs_view0']:.3f}")
        )
    body = "\n".join(f"    {name} & {val} \\\\" for name, val in rows)
    return (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{tab:texture-validation}\n"
        "  \\begin{tabular}{lc}\n"
        "    \\toprule\n"
        "    Metric & Value \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
