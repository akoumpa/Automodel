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

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.optim.config import LRSchedulerConfig, OptimizerConfig
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.optim.utils import build_dion_optimizer, is_dion_optimizer
from nemo_automodel.shared.utils import dtype_from_str

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.distributed.config import DistributedConfig
    from nemo_automodel.components.training.step_scheduler import StepScheduler

logger = logging.getLogger(__name__)


def _fully_shard_megatron_optimizer(model_part: torch.nn.Module, optimizer: torch.optim.Optimizer):
    from nemo_automodel.components.distributed import megatron_fsdp

    if not megatron_fsdp.HAS_MEGATRON_FSDP:
        return optimizer
    return megatron_fsdp.fully_shard_optimizer(model_part, optimizer)


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig | None = None,
    *,
    optimizer_factory: Callable[..., torch.optim.Optimizer] | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    distributed_config: DistributedConfig | None = None,
    device_mesh: DeviceMesh | None = None,
):
    """Build optimizers for a model or its parts.

    ``config`` carries kwargs; ``optimizer_factory`` is always required (the
    component layer does not resolve dotted paths — pass the class directly).
    """
    if config is not None:
        if optimizer_factory is None:
            raise ValueError(
                "When using OptimizerConfig, optimizer_factory must also be provided. "
                "Pass the optimizer class directly (e.g. torch.optim.AdamW)."
            )
        optimizer_kwargs = config.to_kwargs()
    elif optimizer_factory is None:
        raise ValueError("Either config or optimizer_factory must be provided")

    optimizer_kwargs = dict(optimizer_kwargs or {})

    for attr in ("master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        val = optimizer_kwargs.get(attr)
        if isinstance(val, str):
            optimizer_kwargs[attr] = dtype_from_str(val)

    if device_mesh is not None and "tp" in device_mesh.mesh_dim_names and device_mesh["tp"].size() > 1:
        optimizer_kwargs["foreach"] = False  # TP does not support foreach

    has_dion = is_dion_optimizer(optimizer_factory)
    optimizers = []
    for part in getattr(model, "parts", [model]):
        trainable_params = [p for p in part.parameters() if p.requires_grad]
        assert trainable_params, "trainable_params cannot be empty"
        if has_dion:
            opt = build_dion_optimizer(
                optimizer_factory=optimizer_factory,
                optimizer_kwargs=optimizer_kwargs,
                model=part,
                distributed_mesh=device_mesh,
            )
        else:
            opt = optimizer_factory(params=trainable_params, **optimizer_kwargs)
        if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
            assert not has_dion, "Dion optimizer does not support fully_shard_optimizer"
            opt = _fully_shard_megatron_optimizer(part, opt)
        optimizers.append(opt)
    return optimizers


def build_lr_scheduler(
    config: LRSchedulerConfig | None,
    optimizer: list[torch.optim.Optimizer] | torch.optim.Optimizer,
    step_scheduler: StepScheduler,
) -> list[OptimizerParamScheduler] | None:
    """Build LR scheduler(s).  ``None`` config disables scheduling."""
    if config is None:
        return None

    total_steps = (step_scheduler.num_epochs * len(step_scheduler.dataloader)) // step_scheduler.grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    schedulers = []
    for opt in optimizers:
        base_lr = opt.param_groups[0]["lr"]
        base_wd = opt.param_groups[0].get("weight_decay", 0.0)
        schedulers.append(
            OptimizerParamScheduler(
                optimizer=opt,
                init_lr=base_lr * 0.1 if config.init_lr is None else config.init_lr,
                max_lr=base_lr if config.max_lr is None else config.max_lr,
                min_lr=base_lr * 0.01 if config.min_lr is None else config.min_lr,
                lr_warmup_steps=min(1000, total_steps // 10)
                if config.lr_warmup_steps is None
                else config.lr_warmup_steps,
                lr_decay_steps=total_steps if config.lr_decay_steps is None else config.lr_decay_steps,
                lr_decay_style=config.lr_decay_style,
                start_wd=base_wd if config.start_wd is None else config.start_wd,
                end_wd=base_wd if config.end_wd is None else config.end_wd,
                wd_incr_steps=total_steps if config.wd_incr_steps is None else config.wd_incr_steps,
                wd_incr_style=config.wd_incr_style,
                use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
                override_opt_param_scheduler=config.override_opt_param_scheduler,
                wsd_decay_steps=config.wsd_decay_steps,
                lr_wsd_decay_style=config.lr_wsd_decay_style,
            )
        )

    logger.info(
        "Building LR scheduler with total_steps=%d, warmup_steps=%d, decay_style=%s",
        total_steps,
        schedulers[0].lr_warmup_steps,
        config.lr_decay_style,
    )
    return schedulers
