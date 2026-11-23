"""Shared print style for experiment figures."""

from hashlib import sha256

from matplotlib import style
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

MODEL_STYLES = {
    "google/embeddinggemma-2": ("EmbeddingGemma2", "#0072B2", "o"),
    "google/ge-v2": ("Gemini Embedding 2", "#D55E00", "s"),
    "google/siglip2-giant-opt-patch16-384": ("SigLIP2-giant", "#009E73", "^"),
    "jinaai/jina-embeddings-v4": ("Jina Embeddings v4", "#AA4499", "D"),
    "Qwen/Qwen3-VL-Embedding-8B": ("Qwen3-VL-Embedding-8B", "#000000", "v"),
}


def model_style(model_id: str) -> dict[str, str]:
    """Use the same legend label, color and marker in every figure."""
    if model_id in MODEL_STYLES:
        label, color, marker = MODEL_STYLES[model_id]
    else:
        digest = sha256(model_id.encode()).digest()
        label = model_id
        colors = ("#882255", "#117733", "#332288", "#555555")
        markers = ("P", "X", "*", "<")
        color = colors[digest[0] % len(colors)]
        marker = markers[digest[1] % len(markers)]
    return {"label": label, "color": color, "marker": marker}


def dissertation_style():
    """Apply the print settings locally, including during export."""
    print_settings = {
        "figure.figsize": (6.5, 4),
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid.axis": "y",
        "grid.color": "#dedede",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
    return style.context(["seaborn-v0_8-whitegrid", print_settings], after_reset=True)


def accuracy_figure() -> tuple[Figure, Axes]:
    figure = Figure(layout="constrained")
    axes = figure.subplots()
    axes.set_ylabel("Majority-vote validation accuracy")
    axes.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    return figure, axes


def add_model_legend(figure: Figure) -> None:
    figure.legend(loc="outside lower center", ncols=2, frameon=False)
