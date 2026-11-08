"""Метрики кластеризации: силуэт, размеры кластеров. t-SNE визуализация."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

plt.rcParams["figure.dpi"] = 110


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import silhouette_score

    n = len(labels)
    unique = np.unique(labels)
    metrics = {
        "n_clusters": int(len(unique)),
        "n_samples": int(n),
        "sizes": [int((labels == u).sum()) for u in unique],
    }
    if n < 3 or len(unique) < 2 or X.shape[0] < 3 or X.shape[1] < 1:
        metrics["silhouette"] = None
        return metrics
    try:
        metrics["silhouette"] = round(float(silhouette_score(X, labels)), 3)
    except Exception:
        metrics["silhouette"] = None
    return metrics


def tsne_plot(X: np.ndarray, labels: np.ndarray, strategy: str, out_dir: Path,
              titles: list[str] | None = None) -> Path:
    """2D t-SNE эмбеддингов с раскраской по кластерам. Возвращает путь к PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = X.shape[0]
    if n < 3:
        return _placeholder(out_dir, strategy, f"мало данных ({n})")
    perp = min(30.0, max(2.0, (n - 1) / 3.0))
    xy = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X)
    fig, ax = plt.subplots()
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for k, u in enumerate(uniq):
        mask = labels == u
        ax.scatter(xy[mask, 0], xy[mask, 1], s=28, label=f"кластер {u}",
                   color=cmap(k % 10), alpha=0.85)
    if titles:
        for i, t in enumerate(titles):
            ax.annotate(str(i + 1), xy[i], fontsize=7, alpha=0.6)
    ax.set_title(f"t-SNE кластеров: {strategy}")
    ax.legend(fontsize=8, loc="best")
    fp = out_dir / f"tsne_{strategy}.png"
    fig.savefig(fp); plt.close(fig)
    return fp


def _placeholder(out_dir: Path, strategy: str, reason: str) -> Path:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"t-SNE недоступен: {reason}", ha="center", va="center", fontsize=12)
    ax.axis("off")
    fp = out_dir / f"tsne_{strategy}.png"
    fig.savefig(fp); plt.close(fig)
    return fp
