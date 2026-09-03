# Copyright 2026 Alibaba Group Holding Limited
# SPDX-License-Identifier: Apache-2.0

import socket

import pytest

from verl.utils.net_utils import get_free_port, reserve_port


def test_reserve_port_is_an_exclusive_listening_reservation():
    port, reservation = reserve_port("0.0.0.0", [0])
    assert reservation.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1

    contender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        with pytest.raises(OSError):
            contender.bind(("0.0.0.0", port))
    finally:
        contender.close()
        reservation.close()

    successor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    successor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        successor.bind(("127.0.0.1", port))
    finally:
        successor.close()


def test_reserve_port_uses_next_candidate_when_first_is_occupied():
    occupied_port, blocker = reserve_port("0.0.0.0", [0])
    try:
        selected_port, reservation = reserve_port("0.0.0.0", [occupied_port, 0])
        try:
            assert selected_port != occupied_port
        finally:
            reservation.close()
    finally:
        blocker.close()
