#!/usr/bin/env python
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

"""Two-GPU regression for layer-owned FSDP master and compute casting.

Usage:
    torchrun --nproc-per-node=2 tests/functional_tests/training/run_fsdp_casting_ownership.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_with_per_param_compute_dtypes


class _GateParams(nn.Module):
    """Qwen3.5-shaped parameter holder that remains owned by its parent layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.A_log = nn.Parameter(torch.linspace(-0.2, 0.2, hidden_size))
        self.dt_bias = nn.Parameter(torch.linspace(0.1, 0.3, hidden_size))


class _Layer(nn.Module):
    """Tiny layer with ordinary weights and Qwen3.5-shaped gate parameters."""

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self._fp32_params = _GateParams(hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform ``[batch, hidden]`` inputs and return the same shape."""
        for name, parameter in self.named_parameters():
            expected_dtype = torch.float32 if "_fp32_params" in name else torch.bfloat16
            if parameter.dtype is not expected_dtype:
                raise AssertionError(f"forward parameter {name} should be {expected_dtype}, got {parameter.dtype}")
        projected = self.projection(inputs)
        gate = -self._fp32_params.A_log.float().exp() * F.softplus(projected.float() + self._fp32_params.dt_bias)
        return projected.float() * gate


class _ReferenceLayer(_Layer):
    """Independent FP32-master reference with explicit transient BF16 copies."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform ``[batch, hidden]`` inputs and return the same shape."""
        projected = F.linear(inputs, self.projection.weight.to(torch.bfloat16))
        gate = -self._fp32_params.A_log.exp() * F.softplus(projected.float() + self._fp32_params.dt_bias)
        return projected.float() * gate


def _input_for_rank(rank: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(9102 + rank)
    return torch.randn(3, 8, generator=generator, device=device, dtype=torch.bfloat16)


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise AssertionError(f"expected world_size=2, got {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(2026)
    reference = _ReferenceLayer().to(device=device, dtype=torch.float32)
    layer = _Layer().to(device=device, dtype=torch.float32)
    layer.load_state_dict(reference.state_dict())

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    fully_shard_with_per_param_compute_dtypes(
        layer,
        fp32_compute_module_names=("_fp32_params",),
        fully_shard_fn=fully_shard,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        ),
    )

    fsdp_units = [module for module in layer.modules() if isinstance(module, FSDPModule)]
    if fsdp_units != [layer]:
        raise AssertionError(f"expected one parent-owned FSDP unit, got {len(fsdp_units)}")
    # A post-wrap resume reload may replace local tensors; the post-load hook
    # must restore their per-parameter all-gather extensions before first use.
    layer.load_state_dict(layer.state_dict())
    for name, parameter in layer.named_parameters():
        if parameter.dtype is not torch.float32:
            raise AssertionError(f"resident/master parameter {name} should be fp32, got {parameter.dtype}")

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)

    local_inputs = _input_for_rank(rank, device)
    layer(local_inputs).square().mean().backward()
    optimizer.step()

    global_inputs = torch.cat([_input_for_rank(input_rank, device) for input_rank in range(world_size)])
    reference(global_inputs).square().mean().backward()
    reference_optimizer.step()

    reference_params = dict(reference.named_parameters())
    for name, parameter in layer.named_parameters():
        if not isinstance(parameter, DTensor):
            raise AssertionError(f"expected sharded DTensor master for {name}, got {type(parameter)}")
        local_reference = reference_params[name].chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(parameter.to_local(), local_reference, rtol=0, atol=5e-5)

    if rank == 0:
        print("PASS: one FSDP unit, fp32 masters, per-parameter transient compute, optimizer parity")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
