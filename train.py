from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import apply_overrides, load_yaml, save_yaml
from src.data import build_dataloaders
from src.engine import eval_one_epoch, train_one_epoch
from src.loggers import ExperimentLogger
from src.models import build_model
from src.utils import count_parameters, resolve_device, save_checkpoint, set_seed


def build_optimizer(model: nn.Module, optimizer_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Use AdamW for every experiment; only lr and weight_decay vary by config."""
    return optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader_len: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Use cosine annealing for every experiment, stepped once per training batch."""
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs * train_loader_len,
    )


def make_run_dir(config: Dict[str, Any]) -> Path:
    run_cfg = config.get("run", {})
    output_root = Path(run_cfg.get("output_dir", "outputs"))
    run_name = run_cfg.get("name", "experiment")
    if run_cfg.get("use_timestamp", True):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return output_root / f"{run_name}_{stamp}"
    return output_root / run_name


def run_training(config: Dict[str, Any]) -> Dict[str, float]:
    run_cfg = config.get("run", {})
    train_cfg = config.get("train", {})

    run_dir = make_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")

    set_seed(int(run_cfg.get("seed", 42)), deterministic=bool(run_cfg.get("deterministic", False)))
    device = resolve_device(train_cfg.get("device", "auto"))

    logger = ExperimentLogger(config, run_dir)
    logging.info("Run directory: %s", run_dir)
    logging.info("Device: %s", device)

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config.get("model", {})).to(device)
    logging.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    epochs = int(train_cfg.get("epochs", 1))
    scheduler = build_scheduler(optimizer, len(train_loader), epochs)

    if config.get("logging", {}).get("wandb", {}).get("watch_model", False):
        logger.watch(model)

    best_val_accuracy = -1.0
    best_val_epoch = 0
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    last_checkpoint_path = run_dir / "checkpoints" / "last.pt"

    final_metrics: Dict[str, float] = {}
    try:
        for epoch in range(1, epochs + 1):
            train_summary = train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            val_summary = eval_one_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "epoch": epoch,
                "lr": lr,
                "train/loss": train_summary["loss"],
                "train/accuracy": train_summary["accuracy"],
                "val/loss": val_summary["loss"],
                "val/accuracy": val_summary["accuracy"],
                "params/trainable": count_parameters(model),
            }

            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                config,
            )

            if val_summary["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_summary["accuracy"]
                best_val_epoch = epoch
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    metrics,
                    config,
                )

            metrics["best/val_accuracy"] = best_val_accuracy
            metrics["best/epoch"] = best_val_epoch
            logger.log_metrics(metrics, step=epoch)
            final_metrics = metrics

        if config.get("logging", {}).get("wandb", {}).get("log_checkpoints", False):
            logger.log_artifact(best_checkpoint_path, artifact_type="model")

    finally:
        logger.finish()

    return final_metrics


def cli() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models from a YAML config.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Override config values with dotted.key=value syntax.",
    )
    args = parser.parse_args()

    config = apply_overrides(load_yaml(args.config), args.opts)
    run_training(config)


if __name__ == "__main__":
    cli()
