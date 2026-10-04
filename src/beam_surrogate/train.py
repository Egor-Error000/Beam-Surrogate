"""Train the residual MLP on CPU."""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from beam_surrogate.config import SurrogateConfig, load_config
from beam_surrogate.limits import apply_cpu_limit
from beam_surrogate.tracking import tracked_run

logger = logging.getLogger(__name__)

INPUT_DIM = 8
OUTPUT_DIM = 4


def _split_sizes(n: int) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 samples for an 80/10/10 split, got {n}")
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    if val_size < 1:
        val_size = 1
    test_size = n - train_size - val_size
    if test_size < 1:
        train_size -= 1
        test_size = 1
    if train_size < 1:
        raise ValueError(f"Split left no training rows for n={n}")
    return train_size, val_size, test_size


def rotate_batch_multiple(x_batch, y_batch, factor: int = 4):
    """Rotate transverse coordinates and residual targets by a random angle."""
    import torch

    if factor <= 1:
        angles = torch.zeros(x_batch.size(0), 1, device=x_batch.device)
        x_in, y_in = x_batch, y_batch
    else:
        x_in = x_batch.repeat_interleave(factor, dim=0)
        y_in = y_batch.repeat_interleave(factor, dim=0)
        angles = torch.rand(x_in.size(0), 1, device=x_batch.device) * (2 * np.pi)

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x0, y0, xp0, yp0 = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3], x_in[:, 3:4]
    params = x_in[:, 4:]
    x_rotated = torch.cat(
        [
            x0 * cos_a - y0 * sin_a,
            x0 * sin_a + y0 * cos_a,
            xp0 * cos_a - yp0 * sin_a,
            xp0 * sin_a + yp0 * cos_a,
            params,
        ],
        dim=1,
    )
    dx, dy, dxp, dyp = y_in[:, 0:1], y_in[:, 1:2], y_in[:, 2:3], y_in[:, 3:4]
    y_rotated = torch.cat(
        [
            dx * cos_a - dy * sin_a,
            dx * sin_a + dy * cos_a,
            dxp * cos_a - dyp * sin_a,
            dxp * sin_a + dyp * cos_a,
        ],
        dim=1,
    )
    return x_rotated, y_rotated


def run_epoch(model, loader, optimizer, criterion, device, norms, *, is_train: bool, aug_factor: int = 1, calc_phys: bool = False):
    import torch

    from beam_surrogate.metrics import physical_errors

    model.train() if is_train else model.eval()
    epoch_loss = 0.0
    preds, trues, inputs = [], [], []
    with torch.set_grad_enabled(is_train):
        for x_batch, y_batch in loader:
            x_dev = x_batch.to(device)
            y_dev = y_batch.to(device)
            if is_train and aug_factor >= 1:
                x_dev, y_dev = rotate_batch_multiple(x_dev, y_dev, factor=aug_factor)
            if is_train:
                optimizer.zero_grad()
            prediction = model(x_dev)
            loss = criterion(prediction, y_dev)
            if is_train:
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss.item())
            if calc_phys and not is_train:
                preds.append(prediction.detach().cpu())
                trues.append(y_batch.cpu())
                inputs.append(x_batch.cpu())
    avg_loss = epoch_loss / max(len(loader), 1)
    if calc_phys and not is_train:
        metrics = physical_errors(
            torch.cat(preds, dim=0).numpy(),
            torch.cat(trues, dim=0).numpy(),
            torch.cat(inputs, dim=0).numpy(),
            norms,
        )
        return avg_loss, metrics
    return avg_loss, None


def train(cfg: SurrogateConfig, *, track: bool = True) -> dict:
    """Fit ``BeamPredictor`` on CPU and keep the best validation checkpoint."""
    apply_cpu_limit(cfg.max_cpus)
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, random_split

    from beam_surrogate.dataset import BeamDataset, Norms
    from beam_surrogate.model import BeamPredictor

    apply_cpu_limit(cfg.max_cpus)
    device = torch.device("cpu")
    norms = Norms.from_training(cfg.training)
    dataset = BeamDataset(cfg.dataset_path, norms)
    train_size, val_size, test_size = _split_sizes(len(dataset))
    generator = torch.Generator().manual_seed(cfg.training.split_seed)
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    aug = cfg.training.train_aug_factor
    train_batch = max(1, cfg.training.batch_size // aug)
    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    model = BeamPredictor(INPUT_DIM, OUTPUT_DIM, cfg.training.hidden_dims).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.base_lr,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    history: dict[str, list] = {"train_mse": [], "val_mse": [], "mean": [], "median": [], "p95": []}
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with tracked_run("train", cfg, enabled=track) as log:
        log.metric("n_samples", float(len(dataset)))
        log.metric("train_size", float(train_size))
        log.metric("val_size", float(val_size))
        log.metric("test_size", float(test_size))
        _fit_epochs(
            cfg,
            model,
            optimizer,
            criterion,
            scheduler,
            train_loader,
            val_loader,
            device,
            norms,
            aug,
            history,
            log,
        )
        best_val = min(history["val_mse"]) if history["val_mse"] else float("inf")
        wall_s = time.perf_counter() - started
        state = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        test_loss, _ = run_epoch(
            model,
            test_loader,
            None,
            criterion,
            device,
            norms,
            is_train=False,
        )
        history["test_mse"] = test_loss
        history["best_val_mse"] = best_val
        history["wall_time_s"] = wall_s
        cfg.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        log.metric("test_mse", test_loss)
        log.metric("best_val_mse", best_val)
        log.metric("wall_time_s", wall_s)
        log.artifact(cfg.history_path)
        log.artifact(cfg.checkpoint_path)

    logger.info("Test MSE: %.8f | %.2f s | checkpoint %s", test_loss, wall_s, cfg.checkpoint_path)
    return history


def _fit_epochs(
    cfg,
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    val_loader,
    device,
    norms,
    aug,
    history,
    log,
) -> None:
    import torch

    best_val = float("inf")
    for epoch in range(cfg.training.epochs):
        train_loss, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            norms,
            is_train=True,
            aug_factor=aug,
        )
        val_loss, phys = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            norms,
            is_train=False,
            calc_phys=True,
        )
        scheduler.step(val_loss)
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)
        assert phys is not None
        history["mean"].append(phys["mean"])
        history["median"].append(phys["median"])
        history["p95"].append(phys["p95"])
        log.metric("train_mse", train_loss, step=epoch)
        log.metric("val_mse", val_loss, step=epoch)
        log.metric("mean_abs_dr_over_r", phys["mean"], step=epoch)
        log.metric("median_abs_dr_over_r", phys["median"], step=epoch)
        log.metric("p95_abs_dr_over_r", phys["p95"], step=epoch)
        log.metric("r_rms_mm", phys["r_rms_mm"], step=epoch)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.checkpoint_path)
            marker = " best"
        logger.info(
            "Epoch %02d/%d | Train MSE: %.3e | Val MSE: %.3e%s | mean|dr|/r: %.3e | median: %.3e | p95: %.3e | r_rms: %.4f mm",
            epoch + 1,
            cfg.training.epochs,
            train_loss,
            val_loss,
            marker,
            phys["mean"],
            phys["median"],
            phys["p95"],
            phys["r_rms_mm"],
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the beam residual MLP on CPU")
    parser.add_argument("--config", required=True, help="Path to cpu.yaml or smoke.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(load_config(args.config))


if __name__ == "__main__":
    main()
