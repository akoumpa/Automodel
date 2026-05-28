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

"""Public, typed optimizer + LR scheduler configs.

Each config carries its own ``build(...)`` that returns the optimizer (or LR
scheduler).  Runtime objects (model, mesh, optimizer, step scheduler) flow in
through the build args so the config dataclasses stay pure data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.distributed.config import DistributedConfig
    from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
    from nemo_automodel.components.training.step_scheduler import StepScheduler

_RESERVED_FIELDS = frozenset({"name", "extra_kwargs"})

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Dotted-path fallback for optimizers Automodel does not own."""

    name: str = "torch.optim.AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _RESERVED_FIELDS}
        return {**d, **self.extra_kwargs}

    def _resolve_factory(self):
        from importlib import import_module

        module_name, cls_name = self.name.rsplit(".", 1)
        return getattr(import_module(module_name), cls_name)

    def build(
        self,
        model: torch.nn.Module,
        distributed_config: DistributedConfig | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> list[torch.optim.Optimizer]:
        """Build optimizers for ``model.parts`` (or ``[model]``)."""
        return build_optimizer_from_factory(
            self._resolve_factory(), self.to_kwargs(), model, distributed_config, device_mesh
        )


@dataclass
class AdamConfig(OptimizerConfig):
    name: str = "torch.optim.Adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False


@dataclass
class AdamWConfig(OptimizerConfig):
    name: str = "torch.optim.AdamW"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    fused: bool = False


@dataclass
class FusedAdamConfig(OptimizerConfig):
    """``transformer_engine.pytorch.optimizers.FusedAdam``."""

    name: str = "transformer_engine.pytorch.optimizers.FusedAdam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    adam_w_mode: bool = True
    bias_correction: bool = True
    master_weights: bool = True
    master_weight_dtype: str | None = None

    def to_kwargs(self) -> dict[str, Any]:
        d = super().to_kwargs()
        if self.master_weight_dtype is None:
            d.pop("master_weight_dtype", None)
        return d


@dataclass
class FlashAdamWConfig(OptimizerConfig):
    name: str = "flashoptim.FlashAdamW"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    master_weight_bits: int = 24


@dataclass
class MuonConfig(OptimizerConfig):
    """``dion.Muon`` — matrix-aware update for 2D+ params, scalar fallback for 1D."""

    name: str = "dion.Muon"
    lr: float = 5e-4
    weight_decay: float = 0.0
    mu: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    adjust_lr: str = "spectral_norm"
    scalar_opt: str = "adamw"
    scalar_betas: tuple[float, float] = (0.9, 0.999)
    scalar_eps: float = 1e-8


@dataclass
class LRSchedulerConfig:
    """LR scheduler configuration.  ``None`` fields are computed from the training schedule."""

    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    lr_decay_style: str = "cosine"
    init_lr: float | None = None
    max_lr: float | None = None
    min_lr: float | None = None
    start_wd: float | None = None
    end_wd: float | None = None
    wd_incr_steps: int | None = None
    wd_incr_style: str = "constant"
    use_checkpoint_opt_param_scheduler: bool = True
    override_opt_param_scheduler: bool = False
    wsd_decay_steps: int | None = None
    lr_wsd_decay_style: str | None = None

    def build(
        self,
        optimizer: list[torch.optim.Optimizer] | torch.optim.Optimizer,
        step_scheduler: StepScheduler,
    ) -> list[OptimizerParamScheduler]:
        from dataclasses import asdict

        from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler

        total_steps = (step_scheduler.num_epochs * len(step_scheduler.dataloader)) // step_scheduler.grad_acc_steps
        if step_scheduler.max_steps is not None:
            total_steps = min(total_steps, step_scheduler.max_steps)

        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        # Non-None fields on self override the per-optimizer computed defaults below.
        overrides = {k: v for k, v in asdict(self).items() if v is not None}
        schedulers = []
        for opt in optimizers:
            base_lr = opt.param_groups[0]["lr"]
            base_wd = opt.param_groups[0].get("weight_decay", 0.0)
            schedulers.append(
                OptimizerParamScheduler(
                    optimizer=opt,
                    **{
                        "init_lr": base_lr * 0.1,
                        "max_lr": base_lr,
                        "min_lr": base_lr * 0.01,
                        "lr_warmup_steps": min(1000, total_steps // 10),
                        "lr_decay_steps": total_steps,
                        "start_wd": base_wd,
                        "end_wd": base_wd,
                        "wd_incr_steps": total_steps,
                        "wsd_decay_steps": None,
                        "lr_wsd_decay_style": None,
                        **overrides,
                    },
                )
            )
        logger.info(
            "Building LR scheduler with total_steps=%d, warmup_steps=%d, decay_style=%s",
            total_steps,
            schedulers[0].lr_warmup_steps,
            self.lr_decay_style,
        )
        return schedulers


def build_optimizer_from_factory(
    factory,
    kwargs: dict[str, Any],
    model: torch.nn.Module,
    distributed_config: DistributedConfig | None = None,
    device_mesh: DeviceMesh | None = None,
) -> list[torch.optim.Optimizer]:
    """Shared optimizer construction (parts loop + dion + MegatronFSDP)."""
    import torch

    from nemo_automodel.components.distributed.config import MegatronFSDPConfig
    from nemo_automodel.components.optim.utils import build_dion_optimizer, is_dion_optimizer
    from nemo_automodel.shared.utils import dtype_from_str

    kwargs = {
        k: (dtype_from_str(v) if k in {"master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"} and isinstance(v, str) else v)
        for k, v in kwargs.items()
    }
    if device_mesh is not None and "tp" in device_mesh.mesh_dim_names and device_mesh["tp"].size() > 1:
        kwargs["foreach"] = False  # TP does not support foreach

    has_dion = is_dion_optimizer(factory)
    optimizers = []
    for part in getattr(model, "parts", [model]):
        trainable = [p for p in part.parameters() if p.requires_grad]
        assert trainable, "trainable_params cannot be empty"
        opt = (
            build_dion_optimizer(
                optimizer_factory=factory, optimizer_kwargs=kwargs, model=part, distributed_mesh=device_mesh
            )
            if has_dion
            else factory(params=trainable, **kwargs)
        )
        if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
            assert not has_dion, "Dion optimizer does not support fully_shard_optimizer"
            from nemo_automodel.components.distributed import megatron_fsdp

            if megatron_fsdp.HAS_MEGATRON_FSDP:
                opt = megatron_fsdp.fully_shard_optimizer(part, opt)
        optimizers.append(opt)
    return optimizers


__all__ = [
    "AdamConfig",
    "AdamWConfig",
    "FlashAdamWConfig",
    "FusedAdamConfig",
    "LRSchedulerConfig",
    "MuonConfig",
    "OptimizerConfig",
]
