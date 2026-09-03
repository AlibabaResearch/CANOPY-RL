#!/usr/bin/env python3
"""Deterministically route SWE instances across Ray node groups."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable


def _routing_key(instance_id: str) -> str:
    """Return the dataset's repository key embedded in an instance ID."""

    key, separator, _ = instance_id.partition("__")
    if not separator or not key:
        raise ValueError(f"instance_id must contain '<repo>__<task>': {instance_id!r}")
    return key


def get_repo_routing_map(
    instance_id_list: Iterable[str],
    num_nodes: int = 4,
) -> dict[str, list[int]]:
    """Assign busy repositories to one or more least-loaded node groups.

    The expected load of a repository is divided evenly across its assigned
    groups. Individual instances are mapped to those groups by
    :func:`get_assigned_group`.
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    counts = Counter(_routing_key(str(instance_id)) for instance_id in instance_id_list)
    if not counts:
        return {}

    target_average = sum(counts.values()) / num_nodes
    node_loads = [0.0] * num_nodes
    repo_to_nodes: dict[str, list[int]] = {}

    for repo, count in counts.most_common():
        assigned_count = min(num_nodes, max(1, math.ceil(count / target_average)))
        groups = sorted(range(num_nodes), key=lambda group: node_loads[group])[:assigned_count]
        expected_increment = count / assigned_count
        for group in groups:
            node_loads[group] += expected_increment
        repo_to_nodes[repo] = groups

    return repo_to_nodes


def get_assigned_group(
    instance_id: str,
    repo_routing_map: dict[str, list[int]],
    repo: str | None = None,
) -> int:
    """Hash one instance into a group assigned to its repository key."""

    routing_key = repo if repo is not None else _routing_key(instance_id)
    target_nodes = repo_routing_map.get(routing_key)
    if not target_nodes:
        raise KeyError(f"No routing groups were assigned for repository key {routing_key!r}")
    digest = hashlib.md5(instance_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return target_nodes[int(digest, 16) % len(target_nodes)]
