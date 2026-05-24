#!/usr/bin/env python3
"""Generate architecture visualisation PNGs for the PPTX presentation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

BG       = "#0D1B2A"
CARD     = "#132A40"
ACCENT   = "#00B4D8"
ACCENT2  = "#90E0EF"
WHITE    = "#FFFFFF"
LGRAY    = "#CCD6E0"
GREEN    = "#06D6A0"
ORANGE   = "#FFA02F"
RED      = "#EF476F"

OUT = "/Users/randubin/PycharmProjects/PythonProject/MIC2/aws_training"


# ── helpers ───────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, fc=CARD, ec=ACCENT, lw=1.5, radius=0.03):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)

def arrow(ax, x0, y0, x1, y1, color=ACCENT2, lw=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14))

def label(ax, x, y, text, size=10, color=WHITE, ha="center", va="center",
          bold=False, wrap=False):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, wrap=wrap,
            fontfamily="DejaVu Sans")


# ── Figure 1 — Patch Embedding ────────────────────────────────────────────────

def fig_patch_embed():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             facecolor=BG,
                             gridspec_kw={"wspace": 0.05})

    # ── Panel A: mock screenshot ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # fake screenshot elements
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color="#1A3A55"))        # nav
    ax.add_patch(plt.Rectangle((0.02, 0.89), 0.18, 0.08, color=ACCENT,     # logo
                                alpha=0.5))
    ax.text(0.5, 0.93, "⚠  YOUR PC IS INFECTED!", fontsize=9,
            color=RED, ha="center", va="center", fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.add_patch(plt.Rectangle((0.1, 0.55), 0.8, 0.28, color="#1A3A55"))   # alert box
    ax.text(0.5, 0.71, "CRITICAL ALERT", fontsize=8, color=RED,
            ha="center", va="center", fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.text(0.5, 0.63, "Call 1-800-FAKE-NUM", fontsize=7,
            color=LGRAY, ha="center", va="center",
            fontfamily="DejaVu Sans")
    ax.add_patch(plt.Rectangle((0.3, 0.42), 0.4, 0.1,
                                color=RED, alpha=0.8))
    ax.text(0.5, 0.47, "SCAN NOW", fontsize=8, color=WHITE,
            ha="center", va="center", fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.text(0.5, 0.06, "Input\n448 × 448 px", fontsize=9,
            color=ACCENT, ha="center", va="bottom", fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.set_title("Screenshot", color=WHITE, fontsize=11, pad=6)

    # ── Panel B: patch grid ────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.axis("off")

    n = 8  # show 8×8 grid for readability (represents 32×32)
    ps = 1.0 / n
    for r in range(n):
        for c in range(n):
            col = ACCENT if (r == 0 and c == 0) else \
                  "#1E4060" if (r + c) % 2 == 0 else CARD
            ax2.add_patch(plt.Rectangle((c * ps, (n-1-r) * ps), ps, ps,
                                         facecolor=col,
                                         edgecolor="#0D2A3F", linewidth=0.8))

    ax2.text(0.5 * ps, (n - 0.5) * ps, "CLS", fontsize=7,
             color=BG, ha="center", va="center", fontweight="bold",
             fontfamily="DejaVu Sans")
    ax2.add_patch(plt.Rectangle((0, (n-1)*ps), ps, ps,
                                  facecolor=GREEN, edgecolor=WHITE,
                                  linewidth=1.5, zorder=5))
    ax2.text(0.5*ps, (n-0.5)*ps, "CLS", fontsize=7,
             color=BG, ha="center", va="center", fontweight="bold",
             fontfamily="DejaVu Sans", zorder=6)

    ax2.text(0.5, 0.06, f"1024 patch tokens\n+ 1 CLS token  =  1025",
             fontsize=9, color=ACCENT, ha="center", va="bottom",
             fontweight="bold", fontfamily="DejaVu Sans")
    ax2.set_title("Patch Embedding  (32×32 grid)", color=WHITE,
                  fontsize=11, pad=6)

    # ── Panel C: token sequence ────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(BG)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.axis("off")

    tokens = [("[CLS]", GREEN, True)] + \
             [(f"p{i}", ACCENT, False) for i in range(1, 8)] + \
             [("...", LGRAY, False)] + \
             [("p1024", ACCENT, False)]
    tw = 0.08; th = 0.07; gap = 0.005
    cols = 5
    for idx, (name, color, bold) in enumerate(tokens):
        c = idx % cols
        r = idx // cols
        x = 0.04 + c * (tw + gap)
        y = 0.88 - r * (th + 0.035)
        ax3.add_patch(plt.Rectangle((x, y - th), tw, th,
                                     facecolor=CARD if not bold else "#0A3A2A",
                                     edgecolor=color, linewidth=1.5))
        ax3.text(x + tw/2, y - th/2, name, fontsize=7,
                 color=color, ha="center", va="center",
                 fontweight="bold" if bold else "normal",
                 fontfamily="DejaVu Sans")

    ax3.text(0.5, 0.42,
             "Each token = 1024-dim\nfloat vector",
             fontsize=9, color=LGRAY, ha="center", va="center",
             fontfamily="DejaVu Sans")
    ax3.text(0.5, 0.06, "Token sequence fed\ninto transformer",
             fontsize=9, color=ACCENT, ha="center", va="bottom",
             fontweight="bold", fontfamily="DejaVu Sans")
    ax3.set_title("Token Sequence  (1025 × 1024)", color=WHITE,
                  fontsize=11, pad=6)

    # arrow between panels
    for ax_l, ax_r in [(axes[0], axes[1]), (axes[1], axes[2])]:
        fig.patches.append(mpatches.FancyArrowPatch(
            (ax_l.get_position().x1 + 0.005, 0.5),
            (ax_r.get_position().x0 - 0.005, 0.5),
            arrowstyle="-|>", mutation_scale=18,
            color=ACCENT2, lw=2,
            transform=fig.transFigure, zorder=10))

    fig.suptitle("Step 1–2: Image → Patches → Token Sequence",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(f"{OUT}/viz_patch_embed.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("viz_patch_embed.png")


# ── Figure 2 — Full Model Architecture ────────────────────────────────────────

def fig_full_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")

    # ── Image input ────────────────────────────────────────────────────────────
    box(ax, 0.3, 6.2, 1.8, 1.5, fc="#0A2035", ec=ACCENT2, lw=2)
    label(ax, 1.2, 7.2, "Input Image", 10, ACCENT2, bold=True)
    label(ax, 1.2, 6.85, "448 × 448 × 3", 9, LGRAY)
    label(ax, 1.2, 6.55, "(RGB)", 9, LGRAY)

    # ── Patch embed ────────────────────────────────────────────────────────────
    box(ax, 0.3, 4.4, 1.8, 1.5, fc=CARD, ec=ACCENT, lw=2)
    label(ax, 1.2, 5.45, "Patch Embed", 10, ACCENT, bold=True)
    label(ax, 1.2, 5.1, "32×32 patches", 9, LGRAY)
    label(ax, 1.2, 4.75, "+ CLS token", 9, GREEN)
    label(ax, 1.2, 4.5, "→ 1025 × 1024", 8, ACCENT2)

    arrow(ax, 1.2, 6.2, 1.2, 5.9)

    # ── Transformer stack ──────────────────────────────────────────────────────
    n_shown = 4
    block_h = 0.55; block_gap = 0.08
    tx0 = 0.3; tw2 = 4.2
    base_y = 4.35

    label(ax, tx0 + tw2/2, base_y + n_shown * (block_h + block_gap) + 0.55,
          "24× Transformer Blocks  (InternViT-300M)", 11, WHITE, bold=True)

    for i in range(n_shown):
        by = base_y + i * (block_h + block_gap)
        is_lora = (i % 2 == 0)
        box(ax, tx0, by, tw2, block_h, fc=CARD, ec=ACCENT if not is_lora else ORANGE, lw=1.5)
        label(ax, tx0 + tw2/2, by + block_h * 0.65,
              f"Block {i+1}:  Self-Attention (qkv · proj)  +  FFN", 9, LGRAY)
        if is_lora:
            box(ax, tx0 + tw2 - 1.3, by + 0.06, 1.2, block_h - 0.12,
                fc="#2A1A00", ec=ORANGE, lw=1.5)
            label(ax, tx0 + tw2 - 0.7, by + block_h/2,
                  "LoRA\nA·B", 8, ORANGE, bold=True)

    # dots
    dots_y = base_y + n_shown * (block_h + block_gap) + 0.05
    label(ax, tx0 + tw2/2, dots_y, "· · ·  (24 blocks total)  · · ·", 10, LGRAY)

    arrow(ax, 1.2, 4.4, 1.2, 4.3 + n_shown * (block_h + block_gap))

    # horizontal arrow from patch embed to transformer
    arrow(ax, 2.1, 5.15, 0.28, 5.15, color=ACCENT2)

    # ── CLS token extraction ────────────────────────────────────────────────────
    box(ax, 5.1, 5.8, 2.2, 1.0, fc="#0A3A2A", ec=GREEN, lw=2)
    label(ax, 6.2, 6.55, "CLS Token", 11, GREEN, bold=True)
    label(ax, 6.2, 6.15, "Position 0  →  [B, 1024]", 9, LGRAY)
    label(ax, 6.2, 5.9, "Global image summary", 8, ACCENT2)

    arrow(ax, 4.5, 6.1, 5.08, 6.1, color=GREEN)
    label(ax, 4.79, 6.3, "extract\npos 0", 8, LGRAY)

    # ── Dropout ────────────────────────────────────────────────────────────────
    box(ax, 5.1, 4.3, 2.2, 0.9, fc=CARD, ec=ACCENT2, lw=1.5)
    label(ax, 6.2, 4.8, "Dropout", 11, ACCENT2, bold=True)
    label(ax, 6.2, 4.5, "p = 0.1  (training only)", 9, LGRAY)

    arrow(ax, 6.2, 5.8, 6.2, 5.2)

    # ── Linear head ────────────────────────────────────────────────────────────
    box(ax, 5.1, 2.8, 2.2, 0.9, fc=CARD, ec=ORANGE, lw=2)
    label(ax, 6.2, 3.35, "Linear Head", 11, ORANGE, bold=True)
    label(ax, 6.2, 3.0, "1024  →  13", 10, LGRAY)

    arrow(ax, 6.2, 4.3, 6.2, 3.7)

    # ── Logits + classes ────────────────────────────────────────────────────────
    classes = ["Benign", "fake_av", "fin_scam", "mis_offer",
               "fake_app", "tech_supp", "gift_card",
               "forced_n", "susp_vpn", "mal_ext",
               "fake_upd", "blank_LP", "fake_dl"]
    n = len(classes)
    cx0 = 8.5; cy_base = 1.0; cw = 1.05; ch = 0.42; cgap = 0.14
    total_h = n * ch + (n-1) * cgap
    cy_start = (8 - total_h) / 2

    label(ax, cx0 + cw/2, cy_start + total_h + 0.4,
          "Output Logits  [B, 13]", 10, WHITE, bold=True)

    for i, cls in enumerate(classes):
        cy = cy_start + (n - 1 - i) * (ch + cgap)
        color = GREEN if cls == "Benign" else RED if "fake" in cls or "scam" in cls else ACCENT
        box(ax, cx0, cy, cw, ch, fc=CARD, ec=color, lw=1.2)
        label(ax, cx0 + cw/2, cy + ch/2, cls, 7, color)

    # fan arrows from linear to classes
    from_x = 5.1 + 2.2
    from_y = 3.25
    for i in range(n):
        cy = cy_start + (n - 1 - i) * (ch + cgap) + ch / 2
        ax.annotate("", xy=(cx0, cy), xytext=(from_x + 0.5, from_y),
                    arrowprops=dict(arrowstyle="-|>",
                                   color="#334A60", lw=0.8,
                                   mutation_scale=8,
                                   connectionstyle="arc3,rad=0.0"))

    # ── argmax ────────────────────────────────────────────────────────────────
    box(ax, 10.2, 3.0, 1.8, 0.8, fc="#0A3A2A", ec=GREEN, lw=2)
    label(ax, 11.1, 3.4, "argmax", 11, GREEN, bold=True)
    label(ax, 11.1, 3.1, "predicted class", 9, LGRAY)

    arrow(ax, cx0 + cw, 3.6, 10.18, 3.4, color=GREEN)

    # ── LoRA legend ────────────────────────────────────────────────────────────
    box(ax, 5.1, 0.3, 8.7, 1.1, fc="#0D1E2C", ec=ORANGE, lw=1.5)
    label(ax, 5.6, 1.15, "LoRA Adapters  (injected into every attention block):", 9, ORANGE,
          bold=True, ha="left")
    label(ax, 5.6, 0.78, "W  (frozen 300M base)", 9, LGRAY, ha="left")
    label(ax, 7.5, 0.78, "+", 11, WHITE, ha="left")
    label(ax, 7.85, 0.78, "B · A  (trainable: rank 16 × 2 × hidden × 48 layers ≈ 4.7M)", 9,
          ORANGE, ha="left")
    label(ax, 5.6, 0.48, "Only B and A receive gradients — base model stays frozen throughout training",
          8, LGRAY, ha="left")

    fig.suptitle("MIC2 Classifier — Full Model Architecture",
                 color=WHITE, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT}/viz_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("viz_architecture.png")


# ── Figure 3 — Classification Head Detail ─────────────────────────────────────

def fig_classification_head():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14); ax.set_ylim(0, 7)
    ax.axis("off")

    # ── CLS vector ────────────────────────────────────────────────────────────
    n_dims = 20  # show 20 of 1024 dims
    dim_h = 0.22; dim_gap = 0.02
    total_h = n_dims * (dim_h + dim_gap)
    vec_x = 0.4
    cy_start = (7 - total_h) / 2

    label(ax, vec_x + 0.35, cy_start + total_h + 0.35,
          "CLS token\n[1024-dim]", 10, ACCENT, bold=True, ha="center")

    np.random.seed(42)
    vals = np.random.randn(n_dims)
    vmax = np.abs(vals).max()
    for i, v in enumerate(vals):
        cy = cy_start + (n_dims - 1 - i) * (dim_h + dim_gap)
        intensity = abs(v) / vmax
        color = ACCENT if v > 0 else RED
        ax.add_patch(plt.Rectangle((vec_x, cy), 0.7, dim_h,
                                    facecolor=CARD, edgecolor="#1A3A55", lw=0.5))
        bar_w = 0.65 * intensity
        ax.add_patch(plt.Rectangle((vec_x, cy), bar_w, dim_h,
                                    facecolor=color, alpha=0.7))
        if i in (0, 9, 19):
            ax.text(vec_x + 0.75, cy + dim_h/2,
                    f"d{i*51}", fontsize=7, color=LGRAY, va="center",
                    fontfamily="DejaVu Sans")
    ax.text(vec_x + 0.35, cy_start - 0.25, "· · ·  1024 dims", fontsize=8,
            color=LGRAY, ha="center", fontfamily="DejaVu Sans")

    # ── Dropout ────────────────────────────────────────────────────────────────
    box(ax, 2.0, 2.8, 1.6, 1.4, fc=CARD, ec=ACCENT2, lw=1.8)
    label(ax, 2.8, 3.75, "Dropout", 11, ACCENT2, bold=True)
    label(ax, 2.8, 3.4, "p=0.1", 10, LGRAY)
    label(ax, 2.8, 3.05, "training only", 8, LGRAY)
    arrow(ax, 1.1, 3.5, 1.98, 3.5, color=ACCENT2)

    # ── Weight matrix W ────────────────────────────────────────────────────────
    mat_x = 4.0; mat_y = 1.5; mat_w = 2.5; mat_h = 4.0
    box(ax, mat_x, mat_y, mat_w, mat_h, fc="#0A1E30", ec=ORANGE, lw=2)
    label(ax, mat_x + mat_w/2, mat_y + mat_h + 0.25,
          "Weight Matrix  W", 10, ORANGE, bold=True)
    label(ax, mat_x + mat_w/2, mat_y + mat_h - 0.02,
          "13 × 1024", 9, LGRAY)

    # draw mini grid
    rows, cols = 13, 18
    cw2 = mat_w / cols; ch2 = mat_h / rows
    np.random.seed(7)
    wmat = np.random.randn(rows, cols)
    for r in range(rows):
        for c in range(cols):
            v = wmat[r, c]
            alpha = min(abs(v) / 2.0, 0.9)
            color = ORANGE if v > 0 else "#5588AA"
            ax.add_patch(plt.Rectangle(
                (mat_x + c * cw2, mat_y + (rows-1-r) * ch2), cw2, ch2,
                facecolor=color, alpha=alpha, edgecolor="#0A1E30", lw=0.3))

    arrow(ax, 3.6, 3.5, 3.98, 3.5, color=ORANGE)
    label(ax, 3.79, 3.75, "dropout\noutput", 8, LGRAY)

    # ── Class logits ────────────────────────────────────────────────────────────
    classes = [
        ("Benign",              0.02, GREEN),
        ("fake_av",             2.81, RED),
        ("financial_scam",      0.31, ACCENT),
        ("misleading_offers",   0.18, ACCENT),
        ("fake_appstore",       0.09, ACCENT),
        ("tech_support_scam",   3.94, RED),   # ← predicted
        ("gift_card_scan",      0.22, ACCENT),
        ("forced_notification", 0.07, ACCENT),
        ("suspicious_vpn",      0.04, ACCENT),
        ("malicious_extension", 0.11, ACCENT),
        ("fake_updates",        0.08, ACCENT),
        ("blank_LP",            0.03, ACCENT),
        ("fake_downloader",     0.05, ACCENT),
    ]
    n = len(classes)
    lx = 7.3; bar_max_w = 4.5
    ly_start = 0.5
    lh = 0.37; lgap = 0.13
    max_val = max(v for _, v, _ in classes)

    label(ax, lx + bar_max_w/2 + 0.5, ly_start + n*(lh+lgap) + 0.45,
          "Output Logits  →  Softmax  →  Probabilities", 11, WHITE, bold=True)

    for i, (cls, val, color) in enumerate(classes):
        ly = ly_start + (n-1-i) * (lh + lgap)
        predicted = cls == "tech_support_scam"

        # class label
        ec = WHITE if predicted else "#1A3A55"
        ax.add_patch(plt.Rectangle((lx - 1.85, ly), 1.8, lh,
                                    facecolor="#0A2035" if predicted else CARD,
                                    edgecolor=ec, lw=1.0))
        ax.text(lx - 0.98, ly + lh/2, cls, fontsize=8,
                color=WHITE if predicted else LGRAY, ha="center", va="center",
                fontweight="bold" if predicted else "normal",
                fontfamily="DejaVu Sans")

        # bar
        bar_w = (val / max_val) * bar_max_w
        ax.add_patch(plt.Rectangle((lx, ly), bar_max_w, lh,
                                    facecolor="#0A1E30", edgecolor="#1A3A55", lw=0.5))
        ax.add_patch(plt.Rectangle((lx, ly), bar_w, lh,
                                    facecolor=GREEN if predicted else color,
                                    alpha=0.85 if predicted else 0.55))
        ax.text(lx + bar_w + 0.08, ly + lh/2,
                f"{val:.2f}", fontsize=8,
                color=GREEN if predicted else LGRAY, va="center",
                fontweight="bold" if predicted else "normal",
                fontfamily="DejaVu Sans")

        if predicted:
            ax.text(lx + bar_max_w + 0.15, ly + lh/2,
                    "← argmax  ✓", fontsize=9,
                    color=GREEN, va="center", fontweight="bold",
                    fontfamily="DejaVu Sans")

    arrow(ax, mat_x + mat_w, 3.5, lx - 1.87, 3.5, color=ORANGE)

    # ── Cross-entropy loss callout ─────────────────────────────────────────────
    box(ax, 7.3, 6.1, 5.5, 0.65, fc="#0A2A1A", ec=GREEN, lw=1.5)
    label(ax, 10.05, 6.6, "Training: Cross-Entropy Loss", 10, GREEN, bold=True)
    label(ax, 10.05, 6.3,
          "Loss = −log(softmax(logit_true_class))  |  pushes correct class up, others down",
          8, LGRAY)

    fig.suptitle("Classification Head  —  CLS Token → Linear(1024→13) → Predicted Class",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT}/viz_classification_head.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("viz_classification_head.png")


# ── Figure 4 — LoRA Mechanism ─────────────────────────────────────────────────

def fig_lora():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                             gridspec_kw={"wspace": 0.12})

    # ── Panel A: standard vs LoRA weight update ───────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG); ax.set_xlim(0, 7); ax.set_ylim(0, 6); ax.axis("off")

    # Standard fine-tune
    label(ax, 1.7, 5.6, "Standard Fine-tuning", 12, RED, bold=True)
    box(ax, 0.2, 2.5, 3.0, 2.7, fc=CARD, ec=RED, lw=2)
    label(ax, 1.7, 4.55, "W  (full update)", 10, RED, bold=True)
    label(ax, 1.7, 4.1, "D × D", 9, LGRAY)
    # mini matrix
    np.random.seed(1)
    d = 10
    for r in range(d):
        for c in range(d):
            alpha = np.random.uniform(0.3, 0.9)
            ax.add_patch(plt.Rectangle(
                (0.35 + c*0.26, 2.65 + r*0.2), 0.24, 0.18,
                facecolor=RED, alpha=alpha,
                edgecolor="#200000", lw=0.3))
    label(ax, 1.7, 2.55, "ALL D² params updated\n→ huge gradient memory", 9, RED)

    # LoRA
    label(ax, 5.3, 5.6, "LoRA", 12, GREEN, bold=True)
    box(ax, 3.8, 3.5, 2.9, 1.8, fc=CARD, ec="#334A60", lw=1.5)
    label(ax, 5.25, 5.1, "W  (frozen)", 10, LGRAY, bold=True)
    np.random.seed(2)
    for r in range(d):
        for c in range(d):
            ax.add_patch(plt.Rectangle(
                (3.9 + c*0.25, 3.6 + r*0.155), 0.23, 0.14,
                facecolor="#1A3A55", alpha=0.6,
                edgecolor="#0D2A3F", lw=0.2))

    # A matrix (r×D)
    box(ax, 3.8, 1.5, 2.9, 0.8, fc="#1A0A00", ec=ORANGE, lw=2)
    label(ax, 5.25, 2.05, "A  [rank × D]", 10, ORANGE, bold=True)
    label(ax, 5.25, 1.72, "rank = 16", 9, LGRAY)
    np.random.seed(3)
    for r in range(3):
        for c in range(d):
            alpha = np.random.uniform(0.4, 0.9)
            ax.add_patch(plt.Rectangle(
                (3.9 + c*0.25, 1.55 + r*0.2), 0.23, 0.17,
                facecolor=ORANGE, alpha=alpha, edgecolor="#200800", lw=0.2))

    # B matrix (D×r)
    box(ax, 3.8, 0.15, 1.0, 1.2, fc="#1A0A00", ec=ORANGE, lw=2)
    label(ax, 4.3, 0.9, "B", 13, ORANGE, bold=True)
    label(ax, 4.3, 0.55, "[D×r]", 9, LGRAY)
    label(ax, 4.3, 0.3, "init=0", 8, LGRAY)

    ax.annotate("", xy=(5.25, 1.48), xytext=(5.25, 3.48),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                lw=1.5, mutation_scale=12))
    label(ax, 5.75, 2.5, "+", 18, WHITE, bold=True)

    label(ax, 3.5, 0.6, "output =\nW·x + B·A·x", 10, GREEN, bold=True, ha="right")

    label(ax, 1.7, 1.5, "output = W·x + ΔW·x\n(ΔW full D×D updated)", 9, RED, ha="center")

    # ── Panel B: parameter count visual ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG); ax2.set_xlim(0, 7); ax2.set_ylim(0, 6); ax2.axis("off")
    label(ax2, 3.5, 5.7, "Parameter Count Comparison", 12, WHITE, bold=True)

    items = [
        ("Full fine-tune\n300M params", 300, RED, 5.0),
        ("LoRA (rank 16)\n4.7M params", 4.7, GREEN, 3.5),
    ]
    bar_x = 1.8; bar_max_h = 2.8; bar_w2 = 1.4; gap2 = 1.5
    max_val2 = 300
    for i, (name, val, color, label_y) in enumerate(items):
        bx = bar_x + i * (bar_w2 + gap2)
        bh = (val / max_val2) * bar_max_h
        by = 0.8
        ax2.add_patch(plt.Rectangle((bx, by), bar_w2, bar_max_h,
                                     facecolor="#0A1E30", edgecolor="#1A3A55", lw=0.5))
        ax2.add_patch(plt.Rectangle((bx, by), bar_w2, bh,
                                     facecolor=color, alpha=0.8))
        ax2.text(bx + bar_w2/2, by + bh + 0.15, name,
                 fontsize=9, color=color, ha="center", va="bottom",
                 fontweight="bold", fontfamily="DejaVu Sans")

    # ratio callout
    box(ax2, 1.5, 4.0, 4.0, 1.3, fc="#0A2A1A", ec=GREEN, lw=2)
    label(ax2, 3.5, 4.95, "64×  fewer trainable params", 14, GREEN, bold=True)
    label(ax2, 3.5, 4.55, "4.7M / 300M = 1.6%", 11, LGRAY)
    label(ax2, 3.5, 4.2, "near-identical accuracy", 10, ACCENT2)

    # VRAM comparison
    box(ax2, 0.3, 0.15, 6.4, 0.55, fc=CARD, ec=ACCENT2, lw=1.2)
    label(ax2, 3.5, 0.42, "VRAM: full fine-tune > 15 GB   vs   LoRA ~8 GB  (fits A10G, batch=8)", 9, LGRAY)

    fig.suptitle("LoRA: Low-Rank Adaptation — Train Less, Learn the Same",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT}/viz_lora.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("viz_lora.png")


if __name__ == "__main__":
    fig_patch_embed()
    fig_full_architecture()
    fig_classification_head()
    fig_lora()
    print("All images generated.")
