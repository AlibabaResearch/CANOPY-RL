# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Modified by CANOPY contributors in 2026; see patches/verl-canopy.patch.
import errno
import ipaddress
import socket
from collections.abc import Iterable


def is_ipv4(ip_str: str) -> bool:
    """
    Check if the given string is an IPv4 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv6(ip_str: str) -> bool:
    """
    Check if the given string is an IPv6 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str, with_alive_sock: bool = False) -> tuple[int, socket.socket | None]:
    """Find a free port on the given address.

    By default the socket is closed internally, suitable for immediate use.
    Set with_alive_sock=True to keep the bound socket open as a lightweight
    reservation. The caller is responsible for closing the socket before the
    port is actually bound by the target service (e.g. NCCL, uvicorn). Use
    ``reserve_port`` when an exclusive listening reservation is required.
    """
    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, 0))
    port = sock.getsockname()[1]
    if with_alive_sock:
        return port, sock
    sock.close()
    return port, None


def reserve_port(address: str, candidate_ports: Iterable[int]) -> tuple[int, socket.socket]:
    """Exclusively reserve the first available TCP endpoint in ``candidate_ports``.

    Unlike ``get_free_port(..., with_alive_sock=True)``, this helper enters the
    listening state and does not enable ``SO_REUSEADDR``. It is intended for a
    service that must close the reservation immediately before starting the
    real listener.
    """

    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
    attempted: list[int] = []
    last_error: OSError | None = None
    for candidate in candidate_ports:
        if not 0 <= candidate <= 65535:
            raise ValueError(f"Port must be between 0 and 65535, got {candidate}")
        attempted.append(candidate)
        sock = socket.socket(family=family, type=socket.SOCK_STREAM)
        try:
            sock.bind((address, candidate))
            sock.listen(1)
            return sock.getsockname()[1], sock
        except OSError as error:
            sock.close()
            if error.errno != errno.EADDRINUSE:
                raise
            last_error = error

    raise RuntimeError(
        f"No reservable TCP port on {address}; attempted {attempted}"
    ) from last_error
