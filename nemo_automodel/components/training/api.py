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

from dataclasses import asdict
from typing import TYPE_CHECKING

from nemo_automodel.components.training.config import StepSchedulerConfig
from nemo_automodel.components.training.step_scheduler import StepScheduler

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def build_step_scheduler(
    config: StepSchedulerConfig | None,
    dataloader: DataLoader,
    dp_group_size: int,
    local_batch_size: int,
) -> StepScheduler:
    """Build a ``StepScheduler``.  ``None`` config uses defaults."""
    return StepScheduler(
        **asdict(config or StepSchedulerConfig()),
        local_batch_size=local_batch_size,
        dp_size=dp_group_size,
        dataloader=dataloader,
    )
