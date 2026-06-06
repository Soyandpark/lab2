from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.config import flatten_dict, to_jsonable


class ExperimentLogger:
    def __init__(self, config: Dict[str, Any], run_dir: str | Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = None
        self._setup_python_logger()
        self._setup_wandb()

    def _setup_python_logger(self) -> None:
        log_path = self.run_dir / "train.log"
        logging.getLogger().handlers.clear()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )

    def _setup_wandb(self) -> None:
        wandb_cfg = self.config.get("logging", {}).get("wandb", {}) or {}
        if not wandb_cfg.get("enabled", False):
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging is enabled, but wandb is not installed. Run `pip install wandb` "
                "or set logging.wandb.enabled=false."
            ) from exc

        init_kwargs = {
            "project": wandb_cfg.get("project", "lab2-cifar10"),
            "name": self.config.get("run", {}).get("name"),
            "config": to_jsonable(flatten_dict(self.config)),
            "mode": wandb_cfg.get("mode", "online"),
            "dir": str(self.run_dir),
        }
        for optional_key in ["entity", "group", "job_type", "notes", "tags"]:
            value = wandb_cfg.get(optional_key)
            if value:
                init_kwargs[optional_key] = value
        self.wandb_run = wandb.init(**init_kwargs)

    def watch(self, model) -> None:
        if self.wandb_run is not None:
            self.wandb_run.watch(model, log="gradients", log_freq=100)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        message = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        logging.info(message)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None, artifact_type: str = "model") -> None:
        if self.wandb_run is None:
            return
        import wandb

        path = Path(path)
        artifact = wandb.Artifact(name or path.stem, type=artifact_type)
        artifact.add_file(str(path))
        self.wandb_run.log_artifact(artifact)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
