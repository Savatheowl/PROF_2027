"""Семантические эмбеддинги текстов: TF-IDF + SVD (TruncatedSVD). Без e5."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_COMPONENTS = 50


class TextEmbedder:
    """Строит матрицу признаков: TF-IDF по униграммам/биграммам, затем SVD.

    fit() — на корпусе, transform() — на новых текстах (для predict-агента).
    """

    def __init__(self, n_components: int = 32, max_features: int = 20000) -> None:
        self.n_components = min(n_components, MAX_COMPONENTS)
        self.max_features = max_features
        self._tfidf = None
        self._svd = None

    def fit(self, texts: list[str]) -> "TextEmbedder":
        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words=None,
            sublinear_tf=True,
        )
        tfidf = self._tfidf.fit_transform(texts)
        n_components = min(self.n_components, tfidf.shape[0], tfidf.shape[1])
        self._svd = TruncatedSVD(n_components=max(1, n_components), random_state=42)
        self._svd.fit(tfidf)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._tfidf is None or self._svd is None:
            raise RuntimeError("TextEmbedder.fit() не вызывался")
        return self._svd.transform(self._tfidf.transform(texts))

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
