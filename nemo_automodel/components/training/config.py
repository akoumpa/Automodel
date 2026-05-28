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

"""Typed step-scheduler configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from nemo_automodel.components.training.step_scheduler import StepScheduler


@dataclass
class StepSchedulerConfig:
    """YAML-configurable step-scheduler parameters.  Runtime values (dataloader,
    dp_size, local_batch_size) flow in through ``build``."""

    global_batch_size: int = 32
    num_epochs: int | None = 10
    max_steps: int | None = None
    ckpt_every_steps: int | None = 100
    save_checkpoint_every_epoch: bool = True
    val_every_steps: int | None = None
    log_remote_every_steps: int = 1
    gc_every_steps: int | None = None
    start_step: int = 0
    start_epoch: int = 0

    def build(self, dataloader: DataLoader, dp_group_size: int, local_batch_size: int) -> StepScheduler:
        from nemo_automodel.components.training.step_scheduler import StepScheduler

        return StepScheduler(
            **asdict(self),
            local_batch_size=local_batch_size,
            dp_size=dp_group_size,
            dataloader=dataloader,
        )


__all__ = ["StepSchedulerConfig"]
