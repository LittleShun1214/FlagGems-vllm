# Copyright 2026 FlagOS Contributors
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

"""Performance of the Qwen4 QSA fused prepare operator.

The baseline is the upstream vLLM operator, driven with identical inputs.
"""

from __future__ import annotations

import pytest
import torch
from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import qsa_prepare as vllm_qsa_prepare

import flaggems_vllm
from flaggems_vllm.ops.qwen4.qsa_prepare import qwen_qsa_prepare
from tests.test_qwen4_qsa_prepare import (
    COMPRESS_RATIO,
    HEAD_DIM,
    MROPE_SECTION,
    QsaPrepareCase,
)

from . import base

# Each entry is a batch scenario, not a tensor size: _input_fn expands one entry
# into about two dozen operands whose shapes are unrelated to each other, from
# (128,) norm weights up to a (32768, 64) rope table. Only the token count is
# swept because the head counts, head dims, block size and compress ratio are
# fixed by the model.

# Token counts spanning the decode-to-prefill range. 1024 is the threshold
# where the main-attention path switches to token tiling, so the shapes sit on
# both sides of it plus one well above.
SHAPES = [37, 256, 512, 1024, 2048, 4097, 8192]


class Qwen4QsaPrepareBenchmark(base.GenericBenchmark):
    """Use the common runner with the QSA prepare token counts."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(tokens,) for tokens in SHAPES]
        self.shape_desc = "tokens"

    def record_shapes(self, *args, **kwargs):
        """Report the batch layout rather than every operand.

        The kernel takes sixteen tensors plus twenty keyword operands, so the
        default report would print all of their sizes and run past the width of
        the table. These four describe the layout the shape stands for.
        """

        query = args[0]
        return {
            "tokens": int(query.shape[0]),
            "query": list(query.shape),
            "main_qkv": list(kwargs["main_qkv"].shape),
            "main_kv_cache": list(kwargs["main_kv_cache"].shape),
        }


def _input_fn(shape, dtype, device):
    """One single-request batch per token count, buffers allocated once.

    The operator writes its outputs in place, so the buffers are prepared here
    rather than inside the timed callable.
    """

    (tokens,) = shape
    _ = dtype
    case = QsaPrepareCase(
        seq_lens=[tokens],
        query_lens=[tokens],
        history_lens=[0],
        device=device,
    )
    yield (
        *case._launch_args(case.outputs()),
        dict(
            compress_ratio=COMPRESS_RATIO,
            mrope_section=MROPE_SECTION,
            rope_pos_offset=HEAD_DIM,
            **case.main,
        ),
    )


@pytest.mark.qwen4_qsa_prepare
def test_qwen_qsa_prepare():
    """The shipped kernel against the upstream implementation."""

    bench = Qwen4QsaPrepareBenchmark(
        input_fn=_input_fn,
        op_name="qwen_qsa_prepare",
        torch_op=vllm_qsa_prepare,
        gems_op=qwen_qsa_prepare,
        dtypes=[torch.bfloat16],
    )
    bench.run()


pytestmark = pytest.mark.skipif(
    flaggems_vllm.device == "cpu" or not torch.cuda.is_available(),
    reason="Qwen4 QSA prepare benchmark requires an accelerator",
)
