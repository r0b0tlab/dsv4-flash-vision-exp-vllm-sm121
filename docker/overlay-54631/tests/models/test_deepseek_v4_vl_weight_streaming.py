# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.vl_model import (
    DeepseekV4ForConditionalGeneration,
)


class _FakeDeepseekV4Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.finalized_values: list[tuple[float, float]] = []
        self.mhc_finalizations = 0

    def finalize_mega_moe_weights(self) -> None:
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))

    def finalize_mhc_broadcast_weights(self) -> None:
        self.mhc_finalizations += 1


def test_interleaved_vl_weights_stream_and_finalize_once_after_loading() -> None:
    language_model = object.__new__(DeepseekV4ForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = SimpleNamespace(tie_word_embeddings=False)
    language_model.model = _FakeDeepseekV4Model()
    language_model.hf_to_vllm_mapper = WeightsMapper()

    model = object.__new__(DeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = language_model
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    def interleaved_weights():
        yield "language_model.model.tensor_a", torch.tensor([1.0])
        # A full-checkpoint sort would consume the generator before loading
        # the first tensor. This assertion protects one-pass loading.
        assert language_model.model.tensor_a.item() == 1.0
        yield "vision.tensor_b", torch.tensor([2.0])
        assert model.vision.tensor_b.item() == 2.0
        yield "language_model.model.tensor_c", torch.tensor([3.0])

    loaded = model.load_weights(interleaved_weights())

    assert loaded == {
        "language_model.model.tensor_a",
        "vision.tensor_b",
        "language_model.model.tensor_c",
    }
    assert language_model.model.finalized_values == []
    assert language_model.model.mhc_finalizations == 0

    model.process_weights_after_loading()

    assert language_model.model.finalized_values == [(1.0, 3.0)]
    assert language_model.model.mhc_finalizations == 1
