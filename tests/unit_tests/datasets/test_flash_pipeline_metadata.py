# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native NEAT collation preserves the document batch axis before PP splitting."""

import pytest
import torch

from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.models.common.packing import (
    PackingCapabilities,
    configure_packing,
    get_packing_capabilities,
)


@pytest.mark.parametrize("collater", [neat_packed_collater, neat_packed_vlm_collater])
def test_flash_collation_is_batch_major_and_model_entry_builds_microbatch_offsets(collater):
    documents = [
        {
            "input_ids": [2, 3, 4, 5, 0],
            "labels": [3, 4, 5, 6, -100],
            "position_ids": [0, 1, 0, 1, 0],
            "attention_mask": [1, 1, 2, 2, 0],
        },
        {
            "input_ids": [7, 8, 9, 10, 11],
            "labels": [8, 9, 10, 11, 12],
            "position_ids": [0, 0, 1, 2, 3],
            "attention_mask": [1, 2, 2, 2, 2],
        },
    ]
    contract = PackingCapabilities("flash_varlen")
    batch = collater(documents, packing=contract)
    assert "cu_seq_lens_q" not in batch
    assert "attention_mask" not in batch
    # Every tensor keeps the batch axis, so any microbatch split is valid.
    assert all(value.shape[0] == 2 for value in batch.values() if isinstance(value, torch.Tensor))
    torch.testing.assert_close(batch["_packed_seq_ids"], torch.tensor([[1, 1, 2, 2, 0], [1, 2, 2, 2, 2]]))

    class Decoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            """Return metadata for tokens [batch, sequence] without modifying it."""
            return kwargs

    decoder = Decoder()
    configure_packing("flash_attention_2", model=decoder)
    # Whole batch (no PP) and each PP microbatch get offsets for exactly what they see.
    for rows in (slice(0, 2), slice(0, 1), slice(1, 2)):
        ids = batch["_packed_seq_ids"][rows]
        actual = decoder(batch["input_ids"][rows], _packed_seq_ids=ids)
        expected = contract.prepare_attention_inputs(ids)
        for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            torch.testing.assert_close(actual[key], expected[key])
    full = decoder(batch["input_ids"], _packed_seq_ids=batch["_packed_seq_ids"])
    assert full["cu_seq_lens_q"].tolist() == [0, 2, 4, 5, 6, 10]


def test_text_collater_keeps_single_document_boundaries():
    batch = [
        {
            "input_ids": torch.tensor([2, 3, 0]),
            "labels": torch.tensor([3, -100, -100]),
            "position_ids": torch.tensor([0, 1, 0]),
            "attention_mask": torch.tensor([1, 1, 0]),
        }
    ]
    result = neat_packed_collater(batch, packing=get_packing_capabilities("sdpa"))
    assert result["_packed_seq_ids"].tolist() == [[1, 1, 0]]
