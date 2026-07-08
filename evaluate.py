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

    rank_curve = np.zeros(n, dtype=np.int32)

    for _ in range(rank_repeats):
        test_indices = rng.choice(len(attack_traces), size=n, replace=False)
        current_traces = attack_traces[test_indices]
        current_plaintexts = attack_plaintexts[test_indices]

        model.eval()
        with torch.no_grad():
            x = torch.tensor(current_traces, dtype=torch.float32, device=device)
            # log_softmax for numerical stability (avoids log of small probabilities)
            log_probs = torch.log_softmax(model(x), dim=1).cpu().numpy()

        key_scores = np.zeros(256)
        curr_rank_curve = np.zeros(n, dtype=np.int32)

        # accumulate log-likelihood for each key candidate across traces
        for i in range(n):
            pt = int(current_plaintexts[i])
            for candidate in range(256):
                z = leakage(pt, candidate)
                key_scores[candidate] += log_probs[i, z]
            curr_rank_curve[i] = int(np.sum(key_scores > key_scores[true_key_byte]))

        rank_curve += curr_rank_curve
    rank_curve //= rank_repeats

    return rank_curve
