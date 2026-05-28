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

"""Public, typed loss configs.

Each config owns its own ``build()`` — lazy imports keep optional kernel deps
out of module load.  ``LossConfig`` is the dotted-path fallback for losses
Automodel does not own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


@dataclass
class LossConfig:
    """Dotted-path fallback for losses Automodel does not own."""

    name: str = "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> nn.Module:
        module_name, cls_name = self.name.rsplit(".", 1)
        cls = getattr(import_module(module_name), cls_name)
        return cls(**self.extra_kwargs)


@dataclass
class MaskedCrossEntropyConfig(LossConfig):
    name: str = "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    fp32_upcast: bool = True
    ignore_index: int = -100
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        return MaskedCrossEntropy(
            fp32_upcast=self.fp32_upcast,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            **self.extra_kwargs,
        )


@dataclass
class FusedLinearCEConfig(LossConfig):
    name: str = "nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy"
    ignore_index: int = -100
    logit_softcapping: float = 0.0
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        return FusedLinearCrossEntropy(
            ignore_index=self.ignore_index,
            logit_softcapping=self.logit_softcapping,
            reduction=self.reduction,
            **self.extra_kwargs,
        )


@dataclass
class TEParallelCEConfig(LossConfig):
    name: str = "nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy"
    ignore_index: int = -100
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.te_parallel_ce import TEParallelCrossEntropy

        return TEParallelCrossEntropy(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            **self.extra_kwargs,
        )


@dataclass
class KDLossConfig(LossConfig):
    name: str = "nemo_automodel.components.loss.kd_loss.KDLoss"
    ignore_index: int = -100
    temperature: float = 1.0
    fp32_upcast: bool = True

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.kd_loss import KDLoss

        return KDLoss(
            ignore_index=self.ignore_index,
            temperature=self.temperature,
            fp32_upcast=self.fp32_upcast,
            **self.extra_kwargs,
        )


__all__ = [
    "FusedLinearCEConfig",
    "KDLossConfig",
    "LossConfig",
    "MaskedCrossEntropyConfig",
    "TEParallelCEConfig",
]
