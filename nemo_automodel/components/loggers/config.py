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

"""Typed configs for the remote loggers (WandB, MLflow, Comet)."""

from __future__ import annotations

import logging as _logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

_logger = _logging.getLogger(__name__)


@dataclass
class WandbConfig:
    """Forwarded to ``wandb.init()`` (extra kwargs may be added as fields)."""

    project: str = "automodel"
    entity: str | None = None
    name: str = ""
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    save_dir: str | None = None
    notes: str | None = None

    def build(self, run_config: Mapping[str, Any] | None = None, model_name: str | None = None) -> Any:
        """Initialise WandB and return the run."""
        import wandb
        from wandb import Settings

        kwargs = {k: v for k, v in asdict(self).items() if v is not None}
        if kwargs.get("name", "") == "" and model_name:
            kwargs["name"] = "_".join(model_name.split("/")[-2:])
        return wandb.init(
            **kwargs,
            config=dict(run_config) if run_config is not None else None,
            settings=Settings(silent=True),
        )


@dataclass
class MLflowConfig:
    """MLflow run configuration (sidecar resume gated by ``resume``)."""

    experiment_name: str = "automodel-experiment"
    run_name: str = ""
    tracking_uri: str | None = None
    artifact_location: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    resume: bool = True
    description: str | None = None
    flatten_depth: int | None = 1

    def build(self, checkpoint_dir: str | None = None, run_config: Mapping[str, Any] | None = None) -> Any:
        """Initialise MLflow on rank 0; return the run (or ``None`` on other ranks)."""
        import os
        from pathlib import Path

        import torch.distributed as dist

        if not (dist.is_initialized() and dist.get_rank() == 0):
            return None
        try:
            import mlflow
        except ImportError as e:
            raise ImportError("MLflow is not installed. Please install it with: uv add mlflow") from e

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else mlflow.create_experiment(name=self.experiment_name, artifact_location=self.artifact_location)
            )
        except Exception as e:
            _logger.warning(f"Failed to create/get experiment: {e}")
            experiment_id = "0"

        tags = dict(self.tags)
        sidecar = Path(checkpoint_dir) / "mlflow_run_id" if checkpoint_dir else None
        existing_run_id = os.environ.get("MLFLOW_RUN_ID") or (
            sidecar.read_text().strip() if self.resume and sidecar and sidecar.exists() else None
        )
        if self.description is not None:
            tags["mlflow.note.content"] = self.description

        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_id=existing_run_id,
            run_name=self.run_name,
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

                mlflow.log_params(flatten_params_for_mlflow(config_dict, max_depth=self.flatten_depth))
                mlflow.log_dict(config_dict, "config.yaml")
            else:
                from datetime import datetime, timezone

                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                mlflow.log_dict(config_dict, f"config.resumed-{ts}.yaml")

        _logger.info(f"MLflow run started: {run.info.run_id}")
        _logger.info(f"View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
        return run


@dataclass
class CometConfig:
    """Comet ML logger configuration."""

    project_name: str = "automodel"
    workspace: str | None = None
    api_key: str | None = None
    experiment_name: str | None = None
    tags: list[str] = field(default_factory=list)
    auto_metric_logging: bool = False

    def build(self) -> Any:
        """Return a ``CometLogger`` (rank-0 active)."""
        from nemo_automodel.components.loggers.comet_utils import CometLogger

        return CometLogger(**asdict(self))


__all__ = ["CometConfig", "MLflowConfig", "WandbConfig"]
