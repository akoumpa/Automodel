# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Typed wrapper over the YAML ``ConfigNode``.

The YAML→typed coercion happens here, at the recipe input boundary, so the
recipe body only ever sees typed component configs and can call
``self.cfg.<section>.build(...)`` directly.  Unknown attributes fall through
to the raw ``ConfigNode`` for legacy access patterns.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from nemo_automodel.components.distributed.config import DistEnvConfig
from nemo_automodel.components.loggers.config import CometConfig, MLflowConfig, WandbConfig
from nemo_automodel.components.optim.config import LRSchedulerConfig
from nemo_automodel.components.training.config import StepSchedulerConfig

if TYPE_CHECKING:
    from nemo_automodel.components.config.loader import ConfigNode


def _section_kwargs(node: Any) -> dict[str, Any]:
    """Return ``**kwargs`` for a typed config from a ``ConfigNode`` (drops ``_target_``)."""
    d = node.to_dict() if hasattr(node, "to_dict") else dict(node)
    d.pop("_target_", None)
    return d


class RecipeConfig:
    """Typed view over the YAML config.

    Known sections (``dist_env``, ``wandb``, ``mlflow``, ``comet``,
    ``step_scheduler``, ``lr_scheduler``) are exposed as typed dataclass
    instances with ``.build(...)`` methods.  Other attributes delegate to the
    underlying ``ConfigNode``.
    """

    def __init__(self, raw: ConfigNode):
        object.__setattr__(self, "_raw", raw)

    @cached_property
    def dist_env(self) -> DistEnvConfig:
        node = self._raw.get("dist_env", None)
        return DistEnvConfig(**_section_kwargs(node)) if node is not None else DistEnvConfig()

    @cached_property
    def wandb(self) -> WandbConfig | None:
        node = self._raw.get("wandb", None)
        return WandbConfig(**_section_kwargs(node)) if node is not None else None

    @cached_property
    def mlflow(self) -> MLflowConfig | None:
        node = self._raw.get("mlflow", None)
        return MLflowConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def comet(self) -> CometConfig | None:
        node = self._raw.get("comet", None)
        return CometConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def step_scheduler(self) -> StepSchedulerConfig:
        node = self._raw.get("step_scheduler", None)
        if node is None:
            return StepSchedulerConfig()
        kwargs = _section_kwargs(node)
        # local_batch_size is consumed by the dataloader; not part of StepSchedulerConfig.
        kwargs.pop("local_batch_size", None)
        return StepSchedulerConfig(**kwargs)

    @cached_property
    def lr_scheduler(self) -> LRSchedulerConfig | None:
        node = self._raw.get("lr_scheduler", None)
        return LRSchedulerConfig(**_section_kwargs(node)) if node is not None else None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._raw, name)

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._raw.to_dict()

    def to_yaml_dict(self, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self._raw, "to_yaml_dict"):
            return self._raw.to_yaml_dict(**kwargs)
        return self.to_dict()
