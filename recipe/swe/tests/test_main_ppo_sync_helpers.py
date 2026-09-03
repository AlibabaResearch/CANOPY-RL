"""Tests for SWE trainer data-shape helpers."""

# Copyright 2026 Alibaba Group Holding Limited
# SPDX-License-Identifier: Apache-2.0

import torch

from recipe.swe.main_ppo_sync import _pad_nested_or_keep_dense


def test_pad_nested_or_keep_dense_keeps_dense_tensor():
    dense = torch.tensor([[1, 2, 3], [4, 5, 6]])

    result = _pad_nested_or_keep_dense(dense, padding=99)

    assert result is dense


def test_pad_nested_or_keep_dense_pads_nested_tensor():
    nested = torch.nested.nested_tensor(
        [torch.tensor([1, 2, 3]), torch.tensor([4])]
    )

    result = _pad_nested_or_keep_dense(nested, padding=99)

    assert torch.equal(result, torch.tensor([[1, 2, 3], [4, 99, 99]]))
