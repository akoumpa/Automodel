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
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Optional

from torch.distributed.checkpoint.stateful import Stateful

from nemo_automodel.components.training.signal_handler import DistributedSignalHandler

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 9223372036854775807


def _calculate_max_steps(
    num_epochs: Optional[int],
    epoch_len: Optional[int],
    default_max_steps: int = _DEFAULT_MAX_STEPS,
) -> int:
    """Maximum optimizer-step count derived from num_epochs and epoch_len."""
    if epoch_len is None or num_epochs is None:
        return default_max_steps
    return num_epochs * epoch_len


def _calculate_num_epochs(max_steps: Optional[int], epoch_len: Optional[int], default_num_epochs: int = 10) -> int:
    """Number of epochs derived from max_steps and epoch_len."""
    if epoch_len is None or max_steps is None:
        return default_num_epochs
    return ceil(max_steps / epoch_len)


@dataclass
class StepSchedulerConfig:
    """Configuration for :class:`StepScheduler`.

    Attributes:
        global_batch_size: Effective batch per optimizer step across all GPUs.
        local_batch_size: Per-GPU micro-batch size.
        num_epochs: Total number of epochs.  ``None`` -> derived from
            ``max_steps`` and ``epoch_len`` (default 10 if neither known).
        max_steps: Maximum optimizer steps.  ``None`` -> derived from
            ``num_epochs * epoch_len``.
        ckpt_every_steps: Steps between checkpoints.  ``None`` -> once
            per epoch (or half of ``max_steps`` for iterable datasets).
        save_checkpoint_every_epoch: Also save at every epoch boundary.
        val_every_steps: Steps between validation runs.  ``None`` disables.
        log_remote_every_steps: Steps between remote logger calls.
        gc_every_steps: Steps between manual ``gc.collect()`` calls.
            ``None`` disables.
        start_step / start_epoch: Resume counters.
    """

    global_batch_size: int = 32
    local_batch_size: int = 1
    num_epochs: int | None = 10
    max_steps: int | None = None
    ckpt_every_steps: int | None = 100
    save_checkpoint_every_epoch: bool = True
    val_every_steps: int | None = None
    log_remote_every_steps: int = 1
    gc_every_steps: int | None = None
    start_step: int = 0
    start_epoch: int = 0

    def build(
        self,
        dataloader: "DataLoader",
        dp_group_size: int,
        local_batch_size: int | None = None,
    ) -> "StepScheduler":
        """Build a :class:`StepScheduler`.  ``local_batch_size`` overrides the config field."""
        import dataclasses as _dc

        cfg = self if local_batch_size is None else _dc.replace(self, local_batch_size=local_batch_size)
        return StepScheduler(config=cfg, dp_size=dp_group_size, dataloader=dataloader)


class StepScheduler(Stateful):
    """Scheduler tracking gradient accumulation, checkpointing, validation, logging, GC, and step counting."""

    def __init__(
        self,
        config: StepSchedulerConfig,
        dp_size: int,
        dataloader: Optional["DataLoader"],
    ) -> None:
        if config.global_batch_size <= 0:
            raise ValueError(f"global_batch_size must be > 0, got {config.global_batch_size}")
        if config.local_batch_size <= 0:
            raise ValueError(f"local_batch_size must be > 0, got {config.local_batch_size}")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be > 0, got {dp_size}")
        if config.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {config.start_step}")
        if config.start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {config.start_epoch}")

        # ---- pure-config fields ------------------------------------------------
        self.global_batch_size = config.global_batch_size
        self.local_batch_size = config.local_batch_size
        self.dp_size = dp_size
        self.dataloader = dataloader
        self.step = config.start_step
        self.epoch = config.start_epoch
        self.save_checkpoint_every_epoch = config.save_checkpoint_every_epoch

        # ---- derived: gradient accumulation ------------------------------------
        self.grad_acc_steps = max(1, config.global_batch_size // (config.local_batch_size * dp_size))

        # ---- derived: epoch_len (None for IterableDataset) ---------------------
        try:
            self.epoch_len: Optional[int] = ceil(len(dataloader) / self.grad_acc_steps)
        except (NotImplementedError, TypeError, RuntimeError):
            self.epoch_len = None

        # ---- derived: num_epochs / max_steps reconciliation --------------------
        num_epochs = config.num_epochs
        max_steps = config.max_steps
        if num_epochs is None:
            num_epochs = _calculate_num_epochs(max_steps, self.epoch_len)
        if num_epochs is not None and num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0 (or None when max_steps is given), got {num_epochs}")
        self.num_epochs = num_epochs
        self.max_steps = max_steps if max_steps is not None else _calculate_max_steps(num_epochs, self.epoch_len)

        # ---- validation / logging / GC cadences --------------------------------
        self.val_every_steps = config.val_every_steps
        if self.val_every_steps is not None and self.val_every_steps <= 0:
            raise ValueError(f"val_every_steps must be > 0 or None, got {self.val_every_steps}")
        self.log_remote_every_steps = config.log_remote_every_steps
        if self.log_remote_every_steps <= 0:
            raise ValueError(f"log_remote_every_steps must be > 0, got {self.log_remote_every_steps}")
        self.gc_every_steps = config.gc_every_steps
        if self.gc_every_steps is not None and self.gc_every_steps <= 0:
            raise ValueError(f"gc_every_steps must be > 0 or None, got {self.gc_every_steps}")

        # ---- checkpoint cadence default ----------------------------------------
        ckpt_every_steps = config.ckpt_every_steps
        if ckpt_every_steps is None:
            ckpt_every_steps = self.epoch_len if self.epoch_len is not None else max(1, self.max_steps // 2)
            logger.info("ckpt_every_steps not provided; will save every %d steps", ckpt_every_steps)
        if ckpt_every_steps <= 0:
            raise ValueError(f"ckpt_every_steps must be > 0, got {ckpt_every_steps}")
        self.ckpt_every_steps = ckpt_every_steps

        # ---- signal handling ---------------------------------------------------
        self.sig_handler = DistributedSignalHandler().__enter__()
        self.sigterm_flag = False

    def __iter__(self):
        """Iterate over the dataloader, yielding grad-accumulation batch buffers.

        Yields:
            list: a list of ``grad_acc_steps`` micro-batches (possibly shorter at epoch end).
        """
        if self.step >= self.max_steps:
            return
        batch_buffer = []
        for batch in self.dataloader:
            batch_buffer.append(batch)
            if len(batch_buffer) == self.grad_acc_steps:
                yield batch_buffer
                self.step += 1
                batch_buffer = []
                if self.step >= self.max_steps or self.sigterm_flag:
                    return
        if batch_buffer:
            yield batch_buffer
            self.step += 1
        self.epoch += 1

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch on a stateful sampler when present."""
        self.epoch = epoch
        sampler = getattr(self.dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    @property
    def is_remote_logging_step(self) -> bool:
        return self.step % self.log_remote_every_steps == 0

    @property
    def is_val_step(self) -> bool:
        is_val = False
        if self.val_every_steps and self.val_every_steps > 0:
            is_val = self.step % self.val_every_steps == self.val_every_steps - 1
        return (is_val or self.is_ckpt_step) and not self.sigterm_flag

    @property
    def is_ckpt_step(self) -> bool:
        is_ckpt_step = (self.step % self.ckpt_every_steps) == self.ckpt_every_steps - 1
        is_epoch_boundary = self.save_checkpoint_every_epoch and self.is_last_batch
        return is_ckpt_step or is_epoch_boundary or self.is_last_step or self.sigterm_received

    @property
    def is_gc_step(self) -> bool:
        return self.gc_every_steps is not None and self.step % self.gc_every_steps == 0

    @property
    def is_last_step(self) -> bool:
        # +1 because the step is incremented after the batch yield in __iter__'s tail handling
        return self.step + 1 >= self.max_steps

    @property
    def is_last_batch(self) -> bool:
        if self.epoch_len is None:
            return False
        return (self.step % self.epoch_len) == self.epoch_len - 1

    @property
    def sigterm_received(self) -> bool:
        self.sigterm_flag = self.sigterm_flag or any(self.sig_handler.signals_received())
        return self.sigterm_flag

    @property
    def epochs(self):
        """Epoch iterator that respects ``max_steps`` and SIGTERM."""
        for e in range(self.epoch, self.num_epochs):
            if self.step >= self.max_steps or self.sigterm_received:
                return
            yield e

    def state_dict(self) -> dict:
        # At checkpoint time we save step+1 because we yield before incrementing;
        # clamp to max_steps so we don't overshoot if state_dict is called outside the loop.
        return {"step": min(self.max_steps, self.step + 1), "epoch": self.epoch}

    def load_state_dict(self, s: dict) -> None:
        self.step, self.epoch = s["step"], s["epoch"]
