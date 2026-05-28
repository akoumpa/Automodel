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

``OptimizerConfig`` resolves a dotted-path factory for losses Automodel does
not own; typed subclasses expose the full parameter surface for known
optimizers.  Runtime construction lives in ``api.build_optimizer`` because it
needs the model, distributed config, and device mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

_RESERVED_FIELDS = frozenset({"name", "extra_kwargs"})


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


__all__ = [
    "AdamConfig",
    "AdamWConfig",
    "FlashAdamWConfig",
    "FusedAdamConfig",
    "LRSchedulerConfig",
    "MuonConfig",
    "OptimizerConfig",
]
