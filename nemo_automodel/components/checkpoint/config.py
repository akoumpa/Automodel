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

"""Typed checkpoint configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from packaging.version import parse

from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat


def _is_geq_torch_2_9() -> bool:
    return parse(torch.__version__).base_version >= "2.9.0"


@dataclass
class CheckpointingConfig:
    """Configuration for checkpointing."""

    enabled: bool = True
    checkpoint_dir: str | Path = "checkpoints/"
    model_save_format: str = "safetensors"
    model_cache_dir: str | Path | None = None
    model_repo_id: str | None = None
    save_consolidated: bool = True
    is_peft: bool = False
    # copy of the model state dict keys before any parallelization; kept for BW compat.
    model_state_dict_keys: list[str] | None = None
    is_async: bool = False
    dequantize_base_checkpoint: bool | None = None
    original_model_root_dir: str | None = None
    # Parameter prefixes to skip when loading base model.
    skip_task_head_prefixes_for_base_model: list[str] | None = None
    # If True, only rank 0 performs consolidation (needed for remote stores without append).
    single_rank_consolidation: bool = False
    # Optional staging directory for consolidation temp files.
    staging_dir: str | None = None
    # If True, save the original pretrained config.json for transformers v4 compatibility.
    v4_compatible: bool = False
    # If True, use diffusers-compatible index filename for from_pretrained() loading.
    diffusers_compatible: bool = False
    best_metric_key: str = "default"

    def __post_init__(self):
        """Convert a raw string such as "safetensors" into the right Enum."""
        formats = [v.value for v in SerializationFormat]
        assert self.model_save_format in formats, (
            f"Unsupported model save format: {self.model_save_format}. Supported formats: {formats}"
        )
        self.model_save_format = SerializationFormat[self.model_save_format.upper()]
        if self.model_cache_dir is None:
            from huggingface_hub import constants as hf_constants

            self.model_cache_dir = hf_constants.HF_HUB_CACHE
        if self.save_consolidated and not self.v4_compatible:
            logging.warning(
                "save_consolidated=True but v4_compatible=False; "
                "checkpoint assets may be not compatible with transformers v4; "
                "[experimental] set --checkpoint.v4_compatible=True to enable"
            )
        elif self.save_consolidated:
            logging.warning("[experimental] v4_compatible=True enables transformers v4 compatibility")

        if self.is_async and not _is_geq_torch_2_9():
            logging.error("Async mode is only supported for torch >= 2.9.0, disabling async mode")
            self.is_async = False


__all__ = ["CheckpointingConfig"]
