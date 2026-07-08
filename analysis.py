from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


@dataclass
class SNRAccum:
    """Statistics accumulator for SNR = Var(class means) / Mean(class variances)."""

    counts: np.ndarray
    sums: np.ndarray
    sum_sq: np.ndarray

    @staticmethod
    def create(n_features: int, num_classes: int = 256) -> SNRAccum:
        return SNRAccum(
            counts=np.zeros(num_classes, dtype=np.int64),
            sums=np.zeros((num_classes, n_features), dtype=np.float64),
            sum_sq=np.zeros((num_classes, n_features), dtype=np.float64),
        )

    def update(self, traces: np.ndarray, labels: np.ndarray):
        chunk = traces.astype(np.float64) if traces.dtype != np.float64 else traces
        for c in range(self.counts.shape[0]):
            sel = labels == c
            if sel.any():
                self.counts[c] += sel.sum()
                self.sums[c] += chunk[sel].sum(axis=0)
                self.sum_sq[c] += (chunk[sel] ** 2).sum(axis=0)

    def finalize(self) -> np.ndarray:
        """Compute SNR from accumulated statistics. Var computed via E[X^2] - E[X]^2"""
        valid = self.counts > 1
        means = self.sums[valid] / self.counts[valid, None]
        variances = self.sum_sq[valid] / self.counts[valid, None] - means**2
        return np.var(means, axis=0) / np.maximum(np.mean(variances, axis=0), 1e-12)


def plot_snr(
    snrs: dict[str, np.ndarray], xmin: int, xmax: int, save_path: str | None = None
) -> Axes:
    _, ax = plt.subplots(figsize=(14, 5))
    for name, snr in snrs.items():
        normed = snr / (snr.max() + 1e-12)
        ax.plot(normed, label=name, alpha=0.8)
    ax.set_xbound(xmin, xmax)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Normalized SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax
