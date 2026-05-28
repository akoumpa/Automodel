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
from collections.abc import Mapping
from typing import Any

from huggingface_hub import constants as hf_constants

from nemo_automodel.components.checkpoint.config import CheckpointingConfig

logger = logging.getLogger(__name__)


def build_checkpoint_config(
    checkpoint_kwargs: Mapping[str, Any] | None,
    cache_dir: str | None,
    model_repo_id: str | None,
    is_peft: bool,
) -> CheckpointingConfig:
    """Build a ``CheckpointingConfig`` from YAML overrides + runtime info."""
    kwargs = dict(
        enabled=True,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else hf_constants.HF_HUB_CACHE,
        save_consolidated=True,
        is_peft=is_peft,
    )
    user = dict(checkpoint_kwargs) if checkpoint_kwargs is not None else {}
    user.pop("restore_from", None)
    if is_peft and user.get("model_save_format") == "torch_save":
        logger.warning(
            "PEFT checkpointing is not supported for `torch_save`; using safetensors defaults "
            "(preserving `checkpoint_dir` if set)."
        )
        if "checkpoint_dir" in user:
            kwargs["checkpoint_dir"] = user["checkpoint_dir"]
    else:
        kwargs |= user
    return CheckpointingConfig(**kwargs)
