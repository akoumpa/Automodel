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

"""Public, typed step scheduler configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepSchedulerConfig:
    """YAML-configurable step-scheduler parameters.  Runtime values (``dataloader``,
    ``dp_size``, ``local_batch_size``) are passed to ``build_step_scheduler``."""

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


__all__ = ["StepSchedulerConfig"]
