#!/usr/bin/env python3
"""
Compose individual LIWC category PDFs into multi-panel publication figures.

Reads existing per-category PDFs from gender/, age/, pagerank/ directories
and arranges selected categories into themed composite figures.

Per EMNLP_PLAN.md:
  - Main body: fig_age_diffs, fig_pagerank_diffs (2 figures)
  - Supplementary: fig_gender_core, fig_gender_emotion, fig_age_core,
                   fig_age_interactions, fig_pagerank_core (5 figures)
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pypdfium2 as pdfium
import numpy as np

FIGURES_DIR = Path(__file__).parent
OUTPUT_DIR = FIGURES_DIR / "composite"
OUTPUT_DIR.mkdir(exist_ok=True)

# DPI for rendering individual PDFs to images
RENDER_DPI = 300


def pdf_to_image(pdf_path: str | Path) -> np.ndarray:
    """Render first page of a PDF to a numpy RGB array."""
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    bitmap = page.render(scale=RENDER_DPI / 72)
    img = bitmap.to_pil()
    arr = np.asarray(img.convert("RGB"))
    doc.close()
    return arr


def get_pdf_path(category: str, dimension: str) -> Path:
    """Construct the path to an individual LIWC figure PDF."""
    if dimension == "gender":
        return FIGURES_DIR / "gender" / f"{category}_Gender.pdf"
    elif dimension == "age":
        return FIGURES_DIR / "age" / f"{category}_Age5y_LIWC_Heatmap.pdf"
    elif dimension == "pagerank":
        return FIGURES_DIR / "pagerank" / f"{category}_PageRank_Heatmap.pdf"
    elif dimension == "age_diff":
        return FIGURES_DIR / "age_differences" / f"{category}_AgeDiff.pdf"
    elif dimension == "pagerank_diff":
        return FIGURES_DIR / "pagerank_differences" / f"{category}_PageRankDiff.pdf"
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


# ── Human-readable labels for LIWC categories ──────────────────────────
CATEGORY_LABELS = {
    "i": "First-person singular (I)",
    "we": "First-person plural (we)",
    "they": "Third-person plural (they)",
    "shehe": "Third-person singular (she/he)",
    "Tone": "Tone (emotional valence)",
    "tone_pos": "Positive tone",
    "tone_neg": "Negative tone",
    "emo_anger": "Anger",
    "emo_anx": "Anxiety",
    "emo_sad": "Sadness",
    "Analytic": "Analytic thinking",
    "Authentic": "Authenticity",
    "Clout": "Clout (confidence)",
    "power": "Power",
    "conflict": "Conflict",
    "focuspast": "Past focus",
    "focusfuture": "Future focus",
    "focuspresent": "Present focus",
    "prosocial": "Prosocial",
    "moral": "Moral",
    "affiliation": "Affiliation",
    "certitude": "Certitude",
    "Affect": "Affect",
    "emotion": "Emotion",
    "reward": "Reward",
    "risk": "Risk",
}


def get_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


# ── Figure definitions ──────────────────────────────────────────────────
# Each figure is a dict with:
#   name: output filename (without extension)
#   panels: list of (category, dimension) tuples
#   layout: (nrows, ncols)
#   location: "main" (paper body) or "supplementary"

# ── MAIN BODY figures (EMNLP_PLAN.md Section 2) ────────────────────────
# These show the headline findings that REQUIRE directed structure.

MAIN_FIGURES = [
    # Figure N.1: Age-gap decomposition (Findings 1 + 2)
    {
        "name": "fig_age_diffs",
        "panels": [
            ("conflict", "age_diff"),   # (a) Finding 2: conflict inverted-U
            ("Tone", "age_diff"),        # (b) Tone U-shape
            ("Clout", "age_diff"),       # (c) Finding 1: Clout monotonic ↗
            ("power", "age_diff"),       # (d) Finding 1: Power monotonic ↘ (mirror)
            ("affiliation", "age_diff"), # (e) Affiliation gradient
            ("we", "age_diff"),          # (f) We collective S-curve
        ],
        "layout": (2, 3),
        "labels": "abcdef",
        "location": "main",
    },
    # Figure N.2: Prominence-gap decomposition (Findings 3 + 4)
    {
        "name": "fig_pagerank_diffs",
        "panels": [
            ("power", "pagerank_diff"),    # (a) Power V-shape
            ("conflict", "pagerank_diff"), # (b) Finding 3: punching-up asymmetry
            ("Tone", "pagerank_diff"),      # (c) Finding 4: peer solidarity (Tone)
            ("Clout", "pagerank_diff"),     # (d) Clout asymmetric U
            ("i", "pagerank_diff"),         # (e) Finding 4: peer solidarity (I)
            ("we", "pagerank_diff"),        # (f) Finding 4: peer solidarity (we)
        ],
        "layout": (2, 3),
        "labels": "abcdef",
        "location": "main",
    },
]

# ── SUPPLEMENTARY figures ───────────────────────────────────────────────

SUPPLEMENTARY_FIGURES = [
    {
        "name": "fig_gender_core",
        "panels": [
            ("i", "gender"),
            ("Analytic", "gender"),
            ("Authentic", "gender"),
            ("Clout", "gender"),
            ("Tone", "gender"),
            ("power", "gender"),
        ],
        "layout": (2, 3),
        "labels": "abcdef",
        "location": "supplementary",
    },
    {
        "name": "fig_gender_emotion",
        "panels": [
            ("emo_anger", "gender"),
            ("emo_anx", "gender"),
            ("emo_sad", "gender"),
            ("conflict", "gender"),
            ("we", "gender"),
            ("they", "gender"),
        ],
        "layout": (2, 3),
        "labels": "abcdef",
        "location": "supplementary",
    },
    {
        "name": "fig_age_core",
        "panels": [
            ("i", "age"),
            ("Clout", "age"),
            ("focuspast", "age"),
            ("Analytic", "age"),
            ("Tone", "age"),
            ("we", "age"),
        ],
        "layout": (2, 3),
        "labels": "abcdef",
        "location": "supplementary",
    },
    {
        "name": "fig_pagerank_core",
        "panels": [
            ("Tone", "pagerank"),
            ("power", "pagerank"),
            ("i", "pagerank"),
            ("Clout", "pagerank"),
            ("we", "pagerank"),
            ("Analytic", "pagerank"),
        ],
        "layout": (2, 3),
        "labels": "abcdef",
        "location": "supplementary",
    },
    {
        "name": "fig_age_interactions",
        "panels": [
            ("power", "age"),
            ("conflict", "age"),
            ("emo_anger", "age"),
            ("Authentic", "age"),
            ("affiliation", "age"),
            ("Tone", "age"),
        ],
        "layout": (2, 3),
        "labels": "abcdef",
        "location": "supplementary",
    },
]

# Combined list for backward compatibility
FIGURES = MAIN_FIGURES + SUPPLEMENTARY_FIGURES


def compose_figure(fig_spec: dict) -> None:
    """Compose a multi-panel figure from individual PDFs."""
    nrows, ncols = fig_spec["layout"]
    panels = fig_spec["panels"]
    labels = fig_spec.get("labels", "abcdefghijklmnop")
    location = fig_spec.get("location", "supplementary")

    # Render all panel images
    images = []
    for category, dimension in panels:
        pdf_path = get_pdf_path(category, dimension)
        if not pdf_path.exists():
            print(f"  WARNING: {pdf_path} not found, skipping")
            images.append(None)
            continue
        images.append(pdf_to_image(pdf_path))

    # Determine consistent aspect ratio from first valid image
    ref_img = next((img for img in images if img is not None), None)
    if ref_img is None:
        print(f"  ERROR: No panels found for {fig_spec['name']}")
        return
    img_h, img_w = ref_img.shape[:2]
    aspect = img_w / img_h

    # Panel size in inches
    panel_w = 3.5
    panel_h = panel_w / aspect

    fig_w = ncols * panel_w + 0.3  # small margin for labels
    fig_h = nrows * panel_h + 0.15

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    # Larger font for main-body figures (EMNLP_PLAN.md step 8)
    label_fontsize = 14 if location == "main" else 12

    for idx, (ax, img) in enumerate(zip(axes.flat, images)):
        if img is None:
            ax.set_visible(False)
            continue
        ax.imshow(img)
        ax.axis("off")
        # Add panel label
        if idx < len(labels):
            ax.text(
                -0.02, 1.02,
                f"({labels[idx]})",
                transform=ax.transAxes,
                fontsize=label_fontsize,
                fontweight="bold",
                va="bottom",
                ha="right",
            )

    # Hide any unused axes
    for ax in axes.flat[len(images):]:
        ax.set_visible(False)

    plt.subplots_adjust(
        left=0.02, right=0.99, top=0.99, bottom=0.01,
        wspace=0.04, hspace=0.04,
    )

    out_path = OUTPUT_DIR / f"{fig_spec['name']}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out_path} [{location}]")


def main():
    parser = argparse.ArgumentParser(
        description="Compose LIWC multi-panel figures for Quotegraph paper"
    )
    parser.add_argument(
        "--main-only", action="store_true",
        help="Only generate main-body figures (fig_age_diffs, fig_pagerank_diffs)",
    )
    parser.add_argument(
        "--supplementary-only", action="store_true",
        help="Only generate supplementary figures",
    )
    args = parser.parse_args()

    if args.main_only:
        figures = MAIN_FIGURES
    elif args.supplementary_only:
        figures = SUPPLEMENTARY_FIGURES
    else:
        figures = FIGURES

    for fig_spec in figures:
        loc = fig_spec.get("location", "supplementary")
        print(f"Composing {fig_spec['name']} [{loc}]...")
        compose_figure(fig_spec)
    print("Done.")


if __name__ == "__main__":
    main()
