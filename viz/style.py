"""Publication plotting style — restrained scientific palette."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-friendly (Wong) palette — no purple-gradient AI look
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # reddish purple (sparingly)
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

MODEL_COLORS = {
    "TimeKAN": "#0072B2",
    "PlainKAN": "#56B4E9",
    "DLinear": "#E69F00",
    "NLinear": "#F0E442",
    "PatchTST": "#009E73",
    "iTransformer": "#D55E00",
    "Informer": "#CC79A7",
    "Autoformer": "#000000",
    "FEDformer": "#999999",
    "LSTM": "#8B4513",
    "TCN": "#2F4F4F",
    "Naive": "#AAAAAA",
    "ARIMA": "#666666",
}


def apply_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: plt.Figure, path, tight: bool = True):
    path = str(path)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if path.endswith(".png"):
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def color_for(model: str) -> str:
    return MODEL_COLORS.get(model, PALETTE[hash(model) % len(PALETTE)])
