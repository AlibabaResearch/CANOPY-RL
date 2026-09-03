# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from unittest.mock import patch

import pytest
import torch

from verl.utils.megatron.router_replay_patch import RouterReplay, rebuild_router_replay_registry


class _FakeTopKRouter(torch.nn.Module):
    def __init__(self, layer_number, replay=None, is_mtp_layer=False):
        super().__init__()
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.router_replay = replay if replay is not None else RouterReplay()


class _Chunk(torch.nn.Module):
    def __init__(self, routers):
        super().__init__()
        self.routers = torch.nn.ModuleList(routers)


@pytest.fixture(autouse=True)
def _clear_registry():
    RouterReplay.router_instances.clear()
    yield
    RouterReplay.router_instances.clear()


def test_rebuild_drops_orphans_and_uses_live_layer_order():
    with patch("verl.utils.megatron.router_replay_patch.TopKRouter", _FakeTopKRouter):
        stale = [_FakeTopKRouter(1), _FakeTopKRouter(2)]
        live_2 = _FakeTopKRouter(2)
        live_1 = _FakeTopKRouter(1)
        trailing_orphan = _FakeTopKRouter(99)

        info = rebuild_router_replay_registry(_Chunk([live_2, live_1]), expected_layer_numbers=[[1, 2]])

    assert info == {
        "raw_count": 5,
        "active_count": 2,
        "chunk_counts": [2],
        "layer_numbers": [[1, 2]],
    }
    assert RouterReplay.router_instances == [live_1.router_replay, live_2.router_replay]
    assert all(router.router_replay not in RouterReplay.router_instances for router in [*stale, trailing_orphan])


def test_rebuild_preserves_vpp_chunk_order_and_skips_mtp():
    with patch("verl.utils.megatron.router_replay_patch.TopKRouter", _FakeTopKRouter):
        vp0_l2 = _FakeTopKRouter(2)
        vp0_l1 = _FakeTopKRouter(1)
        vp0_mtp = _FakeTopKRouter(3, is_mtp_layer=True)
        vp1_l4 = _FakeTopKRouter(4)
        vp1_l3 = _FakeTopKRouter(3)

        info = rebuild_router_replay_registry(
            [_Chunk([vp0_l2, vp0_mtp, vp0_l1]), _Chunk([vp1_l4, vp1_l3])],
            expected_layer_numbers=[[1, 2], [3, 4]],
        )

    assert info["chunk_counts"] == [2, 2]
    assert info["layer_numbers"] == [[1, 2], [3, 4]]
    assert RouterReplay.router_instances == [
        vp0_l1.router_replay,
        vp0_l2.router_replay,
        vp1_l3.router_replay,
        vp1_l4.router_replay,
    ]


def test_rebuild_rejects_expected_layer_mismatch_even_when_count_matches():
    with patch("verl.utils.megatron.router_replay_patch.TopKRouter", _FakeTopKRouter):
        with pytest.raises(RuntimeError, match="live-layer layout mismatch"):
            rebuild_router_replay_registry(
                _Chunk([_FakeTopKRouter(1), _FakeTopKRouter(3)]), expected_layer_numbers=[[1, 2]]
            )


def test_rebuild_rejects_missing_controller_and_duplicate_ownership():
    with patch("verl.utils.megatron.router_replay_patch.TopKRouter", _FakeTopKRouter):
        missing = _FakeTopKRouter(1)
        missing.router_replay = None
        with pytest.raises(RuntimeError, match="no Verl RouterReplay controller"):
            rebuild_router_replay_registry(_Chunk([missing]), expected_layer_numbers=[[1]])

        shared = RouterReplay()
        with pytest.raises(RuntimeError, match="share one RouterReplay controller"):
            rebuild_router_replay_registry(
                _Chunk([_FakeTopKRouter(1, replay=shared), _FakeTopKRouter(2, replay=shared)]),
                expected_layer_numbers=[[1, 2]],
            )


def test_rebuild_clears_live_controller_state():
    with patch("verl.utils.megatron.router_replay_patch.TopKRouter", _FakeTopKRouter):
        router = _FakeTopKRouter(1)
        router.router_replay.recorded_topk_idx = torch.ones(1)
        router.router_replay.target_topk_idx = torch.ones(1)
        router.router_replay.replay_backward_list = [torch.ones(1)]
        router.router_replay.router_replay_action = object()

        rebuild_router_replay_registry(_Chunk([router]), expected_layer_numbers=[[1]])

    assert router.router_replay.recorded_topk_idx is None
    assert router.router_replay.target_topk_idx is None
    assert router.router_replay.replay_backward_list == []
    assert router.router_replay.router_replay_action is None


@pytest.mark.parametrize("layer_numbers, message", [([None], "no layer_number"), ([1, 1], "duplicate layer numbers")])
def test_rebuild_rejects_ambiguous_layer_order(layer_numbers, message):
    with patch("verl.utils.megatron.router_replay_patch.TopKRouter", _FakeTopKRouter):
        with pytest.raises(RuntimeError, match=message):
            rebuild_router_replay_registry(_Chunk([_FakeTopKRouter(number) for number in layer_numbers]))


def test_rebuild_rejects_empty_tree_and_layout_length_mismatch():
    with patch("verl.utils.megatron.router_replay_patch.TopKRouter", _FakeTopKRouter):
        with pytest.raises(RuntimeError, match="contains no live TopKRouter"):
            rebuild_router_replay_registry(_Chunk([]))
        with pytest.raises(RuntimeError, match="expected-layer layout"):
            rebuild_router_replay_registry([_Chunk([]), _Chunk([])], expected_layer_numbers=[[]])
