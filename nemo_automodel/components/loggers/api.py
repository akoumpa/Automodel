# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builders for remote loggers (WandB, MLflow, Comet)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from nemo_automodel.components.loggers.config import CometConfig, MLflowConfig, WandbConfig


def build_wandb(
    config: WandbConfig,
    run_config: Mapping[str, Any] | None = None,
    model_name: str | None = None,
) -> Any:
    """Initialise WandB and return the run."""
    import wandb
    from wandb import Settings

    kwargs = {k: v for k, v in asdict(config).items() if v is not None}
    if kwargs.get("name", "") == "" and model_name:
        kwargs["name"] = "_".join(model_name.split("/")[-2:])
    return wandb.init(
        **kwargs,
        config=dict(run_config) if run_config is not None else None,
        settings=Settings(silent=True),
    )


def build_comet(config: CometConfig) -> Any:
    """Initialise Comet ML and return the logger (rank-0 only inside ``CometLogger``)."""
    from nemo_automodel.components.loggers.comet_utils import CometLogger

    return CometLogger(**asdict(config))


def build_mlflow(
    config: MLflowConfig,
    checkpoint_dir: str | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> Any:
    """Initialise MLflow on rank 0 and start (or resume) a run.

    Installs a ``sys.excepthook`` so crashed jobs report as FAILED.  Returns
    ``None`` on non-rank-0 processes.
    """
    import logging
    import os
    from pathlib import Path

    import torch.distributed as dist

    _logger = logging.getLogger(__name__)

    if not (dist.is_initialized() and dist.get_rank() == 0):
        return None

    try:
        import mlflow
    except ImportError as e:
        raise ImportError("MLflow is not installed. Please install it with: uv add mlflow") from e

    if config.tracking_uri is not None:
        mlflow.set_tracking_uri(config.tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(name=config.experiment_name, artifact_location=config.artifact_location)
        )
    except Exception as e:
        _logger.warning(f"Failed to create/get experiment: {e}")
        experiment_id = "0"

    tags = dict(config.tags)
    sidecar = Path(checkpoint_dir) / "mlflow_run_id" if checkpoint_dir else None
    existing_run_id = os.environ.get("MLFLOW_RUN_ID") or (
        sidecar.read_text().strip() if config.resume and sidecar and sidecar.exists() else None
    )
    if config.description is not None:
        tags["mlflow.note.content"] = config.description

    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_id=existing_run_id,
        run_name=config.run_name,
        tags=tags,
    )
    if existing_run_id is None and sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(run.info.run_id)

    from nemo_automodel.components.loggers.mlflow_utils import _install_mlflow_failure_hook

    _install_mlflow_failure_hook()

    if run_config is not None:
        config_dict = dict(run_config)
        if existing_run_id is None:
            from nemo_automodel.components.loggers.mlflow_utils import flatten_params_for_mlflow

            mlflow.log_params(flatten_params_for_mlflow(config_dict, max_depth=config.flatten_depth))
            mlflow.log_dict(config_dict, "config.yaml")
        else:
            from datetime import datetime, timezone

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            mlflow.log_dict(config_dict, f"config.resumed-{ts}.yaml")

    _logger.info(f"MLflow run started: {run.info.run_id}")
    _logger.info(f"View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
    return run


__all__ = ["build_comet", "build_mlflow", "build_wandb"]
