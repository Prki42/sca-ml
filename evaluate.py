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
    k: int = 10,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(42)
    n = min(max_traces, len(attack_traces))

    rank_curve = np.zeros(n, dtype=np.int32)

    for _ in range(k):
        test_indicies = rng.choice(len(attack_traces), size=n, replace=False)
        current_traces = attack_traces[test_indicies]
        current_plaintexts = attack_plaintexts[test_indicies]

        model.eval()
        with torch.no_grad():
            x = torch.tensor(current_traces[:n], dtype=torch.float32, device=device)
            scores = model(x)
            log_probs = torch.log_softmax(scores, dim=1).cpu().numpy()

        scores = np.zeros(256)
        curr_rank_curve = np.zeros(n, dtype=np.int32)

        for i in range(n):
            pt = int(current_plaintexts[i])
            for k in range(256):
                z = leakage(pt, k)
                scores[k] += log_probs[i, z]
            curr_rank_curve[i] = int(np.sum(scores > scores[true_key_byte]))

        rank_curve += curr_rank_curve
    rank_curve //= k

    return curr_rank_curve
