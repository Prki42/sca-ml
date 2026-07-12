from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int


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


def save_trained_model(
    result: TrainResult, params: dict, train_params: dict, path: str
):
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "params": params,
            "train_params": train_params,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "best_epoch": result.best_epoch,
        },
        path,
    )


def load_trained_model(path: str, build_fn, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(path, map_location=device)
    model = build_fn(**data["params"])
    model.load_state_dict(data["state_dict"])
    if device:
        model.to(device)
    return model, data


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
    warmup: int = 10,
    device: str | None = None,
    verbose: bool = True,
) -> TrainResult:
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
    best_epoch = 0
    current_epoch = 1
    wait = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    pbar = trange(epochs, desc="Training", disable=not verbose)
    for epoch in pbar:
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            best_epoch = epoch
            wait = 0
        else:
            wait += 1 if current_epoch > warmup else wait
            if wait >= patience:
                pbar.write(f"  early stopping at epoch {epoch + 1}")
                break
        current_epoch += 1

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(model, train_losses, val_losses, best_epoch)
