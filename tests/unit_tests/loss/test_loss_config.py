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

"""Tests for nemo_automodel.components.loss.config — LossConfig hierarchy."""


from nemo_automodel.components.loss.config import (
    FusedLinearCEConfig,
    KDLossConfig,
    LossConfig,
    MaskedCrossEntropyConfig,
    TEParallelCEConfig,
)


class TestMaskedCrossEntropyConfig:
    def test_defaults(self):
        cfg = MaskedCrossEntropyConfig()
        assert cfg.fp32_upcast is True
        assert cfg.ignore_index == -100
        assert cfg.reduction == "sum"


class TestFusedLinearCEConfig:
    def test_defaults(self):
        cfg = FusedLinearCEConfig()
        assert cfg.logit_softcapping == 0.0


class TestTEParallelCEConfig:
    def test_defaults(self):
        cfg = TEParallelCEConfig()
        assert cfg.ignore_index == -100
        assert cfg.reduction == "sum"


class TestKDLossConfig:
    def test_defaults(self):
        cfg = KDLossConfig()
        assert cfg.temperature == 1.0
        assert cfg.fp32_upcast is True


class TestBuild:
    def test_build_masked_ce_from_config(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = MaskedCrossEntropyConfig(fp32_upcast=False).build()
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False

    def test_build_kd_loss_from_config(self):
        from nemo_automodel.components.loss.kd_loss import KDLoss

        loss = KDLossConfig(temperature=2.0).build()
        assert isinstance(loss, KDLoss)
        assert loss.temperature == 2.0

    def test_build_via_loss_config_fallback(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = LossConfig(
            name="nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy",
            extra_kwargs={"fp32_upcast": False},
        ).build()
        assert isinstance(loss, MaskedCrossEntropy)

    def test_build_via_subclass(self):
        from nemo_automodel.components.loss.kd_loss import KDLoss

        loss = KDLossConfig(temperature=3.0).build()
        assert isinstance(loss, KDLoss)
        assert loss.temperature == 3.0
