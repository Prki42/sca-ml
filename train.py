import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def split_profiling(
    traces: np.ndarray,
    labels: np.ndarray,
    plaintexts: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(traces))
    split = int((1 - val_ratio) * len(traces))

    return (
        # train
        traces[idx[:split]],
        labels[idx[:split]],
        # validation
        traces[idx[split:]],
        labels[idx[split:]],
        plaintexts[idx[split:]],
    )


def train_model(
    model: nn.Module,
    traces: np.ndarray,
    labels: np.ndarray,
    val_traces: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    device: str | None = None,
    verbose: bool = True,
) -> nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_ds = TensorDataset(
        torch.from_numpy(traces),
        torch.from_numpy(labels),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = torch.from_numpy(val_traces).to(device)
    val_y = torch.from_numpy(val_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()

        if verbose:
            print(
                f"  epoch {epoch + 1}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
