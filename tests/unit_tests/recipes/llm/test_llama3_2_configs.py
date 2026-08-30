# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import yaml

from nemo_automodel.recipes._dist_utils import parse_distributed_section

REPO_ROOT = Path(__file__).resolve().parents[4]
PEFT_CONFIG = REPO_ROOT / "examples" / "llm_finetune" / "llama3_2" / "llama3_2_1b_squad_peft.yaml"


def test_llama3_2_peft_recipe_uses_spark_safe_memory_and_attention_settings() -> None:
    config = yaml.safe_load(PEFT_CONFIG.read_text(encoding="utf-8"))

    distributed = parse_distributed_section(config["distributed"])

    assert distributed["activation_checkpointing"] is True
    assert config["model"]["force_hf"] is True
    assert config["model"]["attn_implementation"] == "flash_attention_2"
