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

"""Thin recipe-boundary builders.

Recipes still receive ``ConfigNode`` from YAML; these helpers normalise that
into the typed component configs (or ``(callable, kwargs)`` pairs for the
``_target_`` escape hatch) before delegating to the component ``build_*``
functions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from nemo_automodel.components.checkpoint.api import build_checkpoint_config as _build_checkpoint_config
from nemo_automodel.components.loggers.api import build_mlflow as _build_mlflow
from nemo_automodel.components.loggers.api import build_wandb as _build_wandb
from nemo_automodel.components.loggers.config import MLflowConfig, WandbConfig
from nemo_automodel.components.loss.api import build_loss_fn as _build_loss_fn
from nemo_automodel.components.optim.api import build_lr_scheduler as _build_lr_scheduler
from nemo_automodel.components.optim.api import build_optimizer as _build_optimizer
from nemo_automodel.components.optim.config import LRSchedulerConfig
from nemo_automodel.components.training.api import build_step_scheduler as _build_step_scheduler
from nemo_automodel.components.training.config import StepSchedulerConfig


def _as_dict(cfg: Any | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Expected a mapping-like config, got {type(cfg).__name__}")


def _callable_and_kwargs(cfg: Any) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Extract ``(_target_, kwargs)`` from a YAML ConfigNode or mapping."""
    if hasattr(cfg, "to_dict") or isinstance(cfg, Mapping):
        d = _as_dict(cfg)
        target = d.pop("_target_", None)
        if target is not None:
            return target, d
    target = getattr(cfg, "_target_", None)
    if target is not None:
        return target, {}
    if callable(cfg):
        return cfg, {}
    if hasattr(cfg, "instantiate"):
        return cfg.instantiate, {}
    raise AttributeError("Config must provide _target_, be callable, or provide instantiate()")


def build_checkpoint_config(cfg_ckpt: Any, cache_dir: str | None, model_repo_id: str | None, is_peft: bool):
    return _build_checkpoint_config(
        checkpoint_kwargs=_as_dict(cfg_ckpt) if cfg_ckpt is not None else None,
        cache_dir=cache_dir,
        model_repo_id=model_repo_id,
        is_peft=is_peft,
    )


def build_loss_fn(cfg_loss: Any) -> Any:
    factory, kwargs = _callable_and_kwargs(cfg_loss)
    return _build_loss_fn(loss_factory=factory, loss_kwargs=kwargs)


def build_optimizer(model: Any, cfg_opt: Any, distributed_config: Any, device_mesh: Any):
    factory, kwargs = _callable_and_kwargs(cfg_opt)
    return _build_optimizer(
        model=model,
        optimizer_factory=factory,
        optimizer_kwargs=kwargs,
        distributed_config=distributed_config,
        device_mesh=device_mesh,
    )


def build_lr_scheduler(cfg: Any, optimizer: Any, step_scheduler: Any):
    config = None if cfg is None else LRSchedulerConfig(**_as_dict(cfg))
    return _build_lr_scheduler(config=config, optimizer=optimizer, step_scheduler=step_scheduler)


def build_step_scheduler(cfg: Any, dataloader: Any, dp_group_size: int, local_batch_size: int):
    if cfg is None:
        config = None
    else:
        kwargs = _as_dict(cfg)
        assert "_target_" not in kwargs, "_target_ not permitted in step scheduler"
        config = StepSchedulerConfig(**kwargs)
    return _build_step_scheduler(
        config=config,
        dataloader=dataloader,
        dp_group_size=dp_group_size,
        local_batch_size=local_batch_size,
    )


def _model_name_from_cfg(cfg_model: Any) -> str | None:
    name = cfg_model.get("pretrained_model_name_or_path", None)
    if name is not None:
        return name
    nested = cfg_model.get("config", None)
    if nested is None:
        return None
    return nested if isinstance(nested, str) else nested.get("pretrained_model_name_or_path", None)


def build_wandb(cfg: Any):
    model_name = _model_name_from_cfg(cfg.model) if hasattr(cfg, "model") else None
    return _build_wandb(
        config=WandbConfig(**_as_dict(cfg.wandb)),
        run_config=_as_dict(cfg),
        model_name=model_name,
    )


def build_mlflow(cfg: Any):
    mlflow_dict = _as_dict(cfg.mlflow) if hasattr(cfg, "mlflow") and cfg.mlflow else {}
    if not mlflow_dict:
        return None
    run_config = cfg.to_yaml_dict(use_orig_values=True) if hasattr(cfg, "to_yaml_dict") else _as_dict(cfg)
    checkpoint_dir = cfg.get("checkpoint.checkpoint_dir", None) if hasattr(cfg, "get") else None
    return _build_mlflow(
        config=MLflowConfig(**mlflow_dict),
        checkpoint_dir=checkpoint_dir,
        run_config=run_config,
    )
