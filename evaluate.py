from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


def compute_rank_curve(
    model: nn.Module,
    attack_traces: np.ndarray,
    attack_plaintexts: np.ndarray,
    true_key_byte: int,
    leakage: Callable[[int, int], int],
    max_traces: int = 1000,
    rank_repeats: int = 10,  # average over multiple random orderings
    device: str | None = None,
) -> np.ndarray:
    """Rank curve: for each trace count i, how many key candidates score above the true key."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(42)
    n = min(max_traces, len(attack_traces))

    model.eval()
    with torch.no_grad():
        x = torch.tensor(attack_traces, dtype=torch.float32, device=device)
        log_probs = torch.log_softmax(model(x), dim=1).cpu().numpy()

    # Precompute leakage table: (num_traces, 256)
    leakage_table = np.array(
        [[leakage(int(pt), k) for k in range(256)] for pt in attack_plaintexts]
    ).astype(np.int64)

    rank_curve = np.zeros(n, dtype=np.float64)

    for _ in range(rank_repeats):
        idx = rng.choice(len(attack_traces), size=n, replace=False)
        scores = np.take_along_axis(log_probs[idx], leakage_table[idx], axis=1)
        cumulative = np.cumsum(scores, axis=0)
        true_scores = cumulative[:, true_key_byte]
        rank_curve += np.sum(cumulative > true_scores[:, None], axis=1)

    return (rank_curve / rank_repeats).astype(np.int32)
