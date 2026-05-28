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

"""Public, typed configs for the remote loggers (WandB, MLflow, Comet)."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class CometConfig:
    """Comet ML logger configuration."""

    project_name: str = "automodel"
    workspace: str | None = None
    api_key: str | None = None
    experiment_name: str | None = None
    tags: list[str] = field(default_factory=list)
    auto_metric_logging: bool = False


__all__ = ["CometConfig", "MLflowConfig", "WandbConfig"]
