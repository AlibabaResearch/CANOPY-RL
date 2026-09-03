#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Structured, per-step diagnostics for the SWE Agent recipe.

The agent loop records one mutually-exclusive terminal outcome plus zero or more
non-exclusive events for each trajectory.  The PPO driver is the only process
that aggregates and prints these records, which avoids duplicate Ray-worker
logs and keeps tracker metric cardinality bounded.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1

TERMINAL_OUTCOMES = (
    "resolved_full",
    "resolved_partial",
    "unresolved_regression",
    "unresolved_no_progress",
    "rollout_env_init_timeout",
    "rollout_env_init_failed",
    "reset_git_log_failed",
    "rollout_global_timeout",
    "rollout_internal_error",
    "rollout_step_infra_failed",
    "trajectory_output_missing",
    "prompt_too_long",
    "action_timeout_limit",
    "repeated_action_limit",
    "response_length_limit",
    "assistant_turn_limit",
    "patch_empty",
    "agent_terminated_other",
    "eval_env_init_timeout",
    "eval_env_init_failed",
    "patch_apply_timeout",
    "patch_apply_failed",
    "eval_timeout",
    "eval_output_limit",
    "eval_execution_failed",
    "harness_apply_patch_failed",
    "harness_reset_failed",
    "tests_timeout",
    "tests_error",
    "test_markers_missing",
    "test_parser_empty",
    "eval_report_missing",
    "skipped_other",
    "unknown",
)

EVENT_NAMES = (
    "action_attempt",
    "action_parse_success",
    "action_execute_attempt",
    "action_execute_success",
    "action_parse_failed",
    "native_tool_protocol_error",
    "submission_attempt",
    "submission_protocol_rejected",
    "submission_patch_empty_rejected",
    "submission_patch_invalid_rejected",
    "accepted_submission",
    "manual_patch_write_blocked",
    "action_timeout",
    "repeated_action",
    "action_output_limit",
    "action_system_error",
    "action_nonzero_exit",
    "submission_timeout",
    "unknown_event",
)

TIMING_SCALAR_FIELDS = (
    "trajectory_wall_seconds",
    "rollout_env_init_seconds",
    "interaction_seconds",
    "llm_seconds",
    "env_step_seconds",
    "final_eval_seconds",
)

OWNER_NAMES = ("model", "infra", "evaluator", "unknown")

STAGE_NAMES = (
    "rollout_env_init",
    "rollout",
    "prompt",
    "action",
    "submission",
    "eval_env_init",
    "apply_patch",
    "evaluation",
    "tests",
    "unknown",
)

INVALID_EVAL_OUTCOMES = frozenset(
    {
        "eval_env_init_timeout",
        "eval_env_init_failed",
        "patch_apply_timeout",
        "patch_apply_failed",
        "eval_timeout",
        "eval_output_limit",
        "eval_execution_failed",
        "harness_apply_patch_failed",
        "harness_reset_failed",
        "tests_timeout",
        "tests_error",
        "test_markers_missing",
        "test_parser_empty",
        "eval_report_missing",
    }
)

INFRA_FAILURE_OUTCOMES = frozenset(
    {
        "rollout_env_init_timeout",
        "rollout_env_init_failed",
        "reset_git_log_failed",
        "rollout_internal_error",
        "rollout_step_infra_failed",
        "trajectory_output_missing",
        "eval_env_init_timeout",
        "eval_env_init_failed",
        "eval_execution_failed",
        "harness_reset_failed",
    }
)

SET_TERMINATION_OUTCOMES = frozenset(
    {"response_length_limit", "assistant_turn_limit", "agent_terminated_other"}
)
REAL_EVAL_OUTCOMES = frozenset(
    {"resolved_full", "resolved_partial", "unresolved_regression", "unresolved_no_progress"}
)
SET_APPLY_FAILURE_OUTCOMES = frozenset(
    {"patch_apply_timeout", "patch_apply_failed", "harness_apply_patch_failed"}
)
TRACKER_SET_EVAL_OUTCOMES = frozenset(
    {
        "response_length_limit",
        "assistant_turn_limit",
        "agent_terminated_other",
        "patch_empty",
        "action_timeout_limit",
        "repeated_action_limit",
    }
) | SET_APPLY_FAILURE_OUTCOMES
NONFAKE_INVALID_EVAL_OUTCOMES = INVALID_EVAL_OUTCOMES - SET_APPLY_FAILURE_OUTCOMES
FAKE_TASK_SKIPPED_OUTCOMES = frozenset({"prompt_too_long", "skipped_other"})
FAKE_GLOBAL_TIMEOUT_OUTCOMES = frozenset({"rollout_global_timeout"})
FAKE_ROLLOUT_ENV_OUTCOMES = frozenset(
    {"rollout_env_init_timeout", "rollout_env_init_failed"}
)
FAKE_EVAL_ENV_OUTCOMES = frozenset({"eval_env_init_timeout", "eval_env_init_failed"})
FAKE_EVAL_TIMEOUT_OUTCOMES = frozenset({"eval_timeout", "tests_timeout"})
FAKE_EVAL_ERROR_OUTCOMES = frozenset(
    {
        "eval_output_limit",
        "eval_execution_failed",
        "harness_apply_patch_failed",
        "harness_reset_failed",
        "tests_error",
        "test_markers_missing",
        "test_parser_empty",
        "eval_report_missing",
    }
)


def _safe_finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def new_trajectory_diagnostics(enabled: bool, phase: str) -> dict[str, Any] | None:
    """Return a fresh trajectory-local record, or ``None`` when disabled."""

    if not enabled:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "core_metrics_version": 1,
        "action_metrics_version": 2,
        "timing_metrics_version": 1,
        "phase": phase,
        "terminal_outcome": "unknown",
        "final_stage": "unknown",
        "failure_owner": "unknown",
        "is_fake": False,
        "did_real_eval": False,
        "agent_state": "unknown",
        "failure_detail": "",
        "events": {},
        "timing": {},
        "patch_length": 0,
        "reward_score": None,
    }


def record_event(diagnostics: dict[str, Any] | None, event: str, count: int = 1) -> None:
    """Increment a non-exclusive event counter on a trajectory record."""

    if diagnostics is None or count <= 0:
        return
    event = event if event in EVENT_NAMES else "unknown_event"
    events = diagnostics.setdefault("events", {})
    events[event] = int(events.get(event, 0)) + int(count)


def record_timing(
    diagnostics: dict[str, Any] | None,
    name: str,
    seconds: float,
    *,
    accumulate: bool = False,
) -> None:
    """Record one finite, non-negative trajectory timing in seconds."""

    if diagnostics is None or name not in TIMING_SCALAR_FIELDS:
        return
    value = _safe_finite_float(seconds, default=-1.0)
    if value < 0.0:
        return
    timing = diagnostics.setdefault("timing", {})
    if accumulate:
        value += _safe_finite_float(timing.get(name, 0.0), default=0.0)
    timing[name] = float(value)


def record_action_timing(
    diagnostics: dict[str, Any] | None, seconds: float
) -> None:
    """Record one dispatched Action's caller-observed ENV RPC wall time."""

    if diagnostics is None:
        return
    value = _safe_finite_float(seconds, default=-1.0)
    if value < 0.0:
        return
    timing = diagnostics.setdefault("timing", {})
    timing["env_step_seconds"] = float(
        _safe_finite_float(timing.get("env_step_seconds", 0.0)) + value
    )
    timing["env_step_timed_count"] = int(
        timing.get("env_step_timed_count", 0)
    ) + 1


def set_terminal_outcome(
    diagnostics: dict[str, Any] | None,
    outcome: str,
    *,
    stage: str,
    owner: str,
    detail: str = "",
    overwrite: bool = False,
) -> None:
    """Set the one terminal outcome, preserving an earlier specific cause."""

    if diagnostics is None:
        return
    current = diagnostics.get("terminal_outcome", "unknown")
    if current != "unknown" and not overwrite:
        return
    diagnostics["terminal_outcome"] = outcome if outcome in TERMINAL_OUTCOMES else "unknown"
    diagnostics["final_stage"] = stage
    diagnostics["failure_owner"] = owner
    diagnostics["failure_detail"] = str(detail or "")[:1000]


def finalize_trajectory_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    report: dict[str, Any] | None,
    agent_state: str,
    did_real_eval: bool,
    is_fake: bool,
    request_id: str,
    task_id: str,
    patch_length: int,
    reward_score: float | None,
) -> dict[str, Any] | None:
    """Finalize a record and derive a test outcome when no earlier cause exists."""

    if diagnostics is None:
        return None
    report = report if isinstance(report, dict) else {}
    diagnostics["agent_state"] = agent_state
    diagnostics["is_fake"] = bool(is_fake)
    diagnostics["request_id"] = str(request_id)
    diagnostics["task_id"] = str(task_id)
    diagnostics["patch_length"] = max(int(patch_length), 0)
    try:
        diagnostics["reward_score"] = float(reward_score) if reward_score is not None else None
    except (TypeError, ValueError):
        diagnostics["reward_score"] = None

    if diagnostics.get("terminal_outcome", "unknown") == "unknown":
        eval_failure_code = str(report.get("eval_failure_code", "") or "")
        failure_code_to_outcome = {
            "harness_apply_patch_failed": "harness_apply_patch_failed",
            "harness_reset_failed": "harness_reset_failed",
            "tests_timeout": "tests_timeout",
            "tests_error": "tests_error",
            "test_markers_missing": "test_markers_missing",
            "test_parser_empty": "test_parser_empty",
            "eval_report_missing": "eval_report_missing",
        }
        if eval_failure_code in failure_code_to_outcome:
            set_terminal_outcome(
                diagnostics,
                failure_code_to_outcome[eval_failure_code],
                stage="evaluation",
                owner="evaluator",
                detail=report.get("msg", eval_failure_code),
            )
        elif did_real_eval:
            f2p = _safe_finite_float(report.get("f2p_rate", 0.0))
            p2p = _safe_finite_float(report.get("p2p_rate", 0.0))
            resolved = bool(report.get("resolved", False))
            if resolved or (f2p >= 1.0 and p2p >= 1.0):
                outcome = "resolved_full"
            elif f2p > 0.0 and p2p >= 1.0:
                outcome = "resolved_partial"
            elif p2p < 1.0:
                outcome = "unresolved_regression"
            else:
                outcome = "unresolved_no_progress"
            set_terminal_outcome(
                diagnostics,
                outcome,
                stage="tests",
                owner="model",
                detail=report.get("msg", "evaluated"),
            )

    outcome = diagnostics.get("terminal_outcome", "unknown")
    diagnostics["invalid_eval"] = outcome in INVALID_EVAL_OUTCOMES
    diagnostics["infra_failure"] = outcome in INFRA_FAILURE_OUTCOMES
    diagnostics["did_real_eval"] = bool(did_real_eval and not diagnostics["invalid_eval"])
    diagnostics["events"] = {
        str(name): int(value)
        for name, value in diagnostics.get("events", {}).items()
        if int(value) > 0
    }
    return diagnostics


def to_python_list(value: Any) -> list[Any]:
    """Convert TQ non-tensor containers, including ``LinkedList``, safely."""

    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    converted = tolist() if callable(tolist) else value
    return converted if isinstance(converted, list) else list(converted)


def select_final_session_indices(
    keys: Sequence[str], padding_flags: Sequence[bool] | None = None
) -> list[int]:
    """Return only each non-padding session's last AgentLoop output."""

    if padding_flags is None:
        padding_flags = [False] * len(keys)
    if len(keys) != len(padding_flags):
        raise ValueError("keys and padding_flags must have the same length")

    final_by_session: dict[str, tuple[int, int]] = {}
    for pos, (key, is_padding) in enumerate(zip(keys, padding_flags, strict=True)):
        if is_padding:
            continue
        parts = str(key).rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            session_key = f"{parts[0]}_{parts[1]}"
            output_index = int(parts[2])
        else:
            session_key = str(key)
            output_index = 0
        previous = final_by_session.get(session_key)
        if previous is None or output_index > previous[0]:
            final_by_session[session_key] = (output_index, pos)
    return [pos for _, pos in sorted(final_by_session.values(), key=lambda item: item[1])]


def _normalized_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        return {
            "schema_version": SCHEMA_VERSION,
            "terminal_outcome": "unknown",
            "events": {},
            "is_fake": False,
            "did_real_eval": False,
            "invalid_eval": False,
            "infra_failure": False,
            "task_id": "unknown",
            "request_id": "unknown",
            "reward_score": None,
            "failure_detail": "diagnostics missing or invalid",
            "timing": {},
            "_timing_instrumented": False,
            "_diagnostics_missing": True,
        }
    outcome = record.get("terminal_outcome", "unknown")
    if outcome not in TERMINAL_OUTCOMES:
        outcome = "unknown"
    normalized = dict(record)
    normalized["terminal_outcome"] = outcome
    normalized_events: Counter[str] = Counter()
    for name, value in record.get("events", {}).items():
        value = int(value)
        if value <= 0:
            continue
        normalized_events[name if name in EVENT_NAMES else "unknown_event"] += value
    normalized["events"] = dict(normalized_events)
    owner = str(normalized.get("failure_owner", "unknown"))
    normalized["failure_owner"] = owner if owner in OWNER_NAMES else "unknown"
    stage = str(normalized.get("final_stage", "unknown"))
    normalized["final_stage"] = stage if stage in STAGE_NAMES else "unknown"
    try:
        normalized["reward_score"] = (
            float(normalized["reward_score"])
            if normalized.get("reward_score") is not None
            else None
        )
    except (TypeError, ValueError):
        normalized["reward_score"] = None
    normalized_timing: dict[str, float | int] = {}
    timing_instrumented = record.get("timing_metrics_version") == 1
    raw_timing = record.get("timing", {})
    if timing_instrumented and isinstance(raw_timing, dict):
        for name in TIMING_SCALAR_FIELDS:
            if name not in raw_timing:
                continue
            value = _safe_finite_float(raw_timing[name], default=-1.0)
            if value >= 0.0:
                normalized_timing[name] = float(value)
        try:
            timed_count = int(raw_timing.get("env_step_timed_count", 0))
        except (TypeError, ValueError):
            timed_count = 0
        if timed_count >= 0:
            normalized_timing["env_step_timed_count"] = timed_count
    normalized["timing"] = normalized_timing
    normalized["_timing_instrumented"] = bool(timing_instrumented)
    normalized["_diagnostics_missing"] = False
    return normalized


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a deterministic linear percentile, or ``None`` for no samples."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _timing_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "sample_count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "sample_count": len(values),
        "mean": float(math.fsum(values) / len(values)),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def _build_timing_summary(
    records: Sequence[dict[str, Any]], *, action_dispatched: int
) -> dict[str, Any]:
    instrumented_records = sum(
        bool(record.get("_timing_instrumented", False)) for record in records
    )
    fields: dict[str, dict[str, float | int | None]] = {}
    for name in TIMING_SCALAR_FIELDS:
        values = [
            float(record["timing"][name])
            for record in records
            if name in record.get("timing", {})
        ]
        fields[name] = _timing_stats(values)

    timed_action_count = sum(
        int(record.get("timing", {}).get("env_step_timed_count", 0))
        for record in records
    )
    env_step_sum_seconds = math.fsum(
        float(record.get("timing", {}).get("env_step_seconds", 0.0))
        for record in records
        if int(record.get("timing", {}).get("env_step_timed_count", 0)) > 0
    )
    if timed_action_count > action_dispatched:
        raise AssertionError("timed Action calls cannot exceed dispatched Action calls")
    return {
        "version": 1,
        "instrumented_records": int(instrumented_records),
        "coverage_rate_of_trajectories": _rate(instrumented_records, len(records)),
        "fields": fields,
        "action": {
            "timed_count": int(timed_action_count),
            "sum_seconds": float(env_step_sum_seconds),
            "mean_seconds": (
                float(env_step_sum_seconds / timed_action_count)
                if timed_action_count
                else None
            ),
            "coverage_rate_of_dispatched": _rate(
                timed_action_count, action_dispatched
            ),
        },
    }


def _build_core_summary(
    records: Sequence[dict[str, Any]],
    event_occurrences: Counter[str],
    *,
    expected_trajectories: int | None,
    missing_outputs: int,
    excess_outputs: int,
    unexpected_outputs: int,
) -> dict[str, Any]:
    """Build the small, mutually-exclusive user-facing diagnostic view."""

    counts: Counter[str] = Counter()
    counts["total"] = len(records)
    conflicts = 0
    instrumented_records = sum(
        not record.get("_diagnostics_missing", False)
        and record.get("core_metrics_version") == 1
        for record in records
    )
    core_metrics_available = instrumented_records == len(records)
    action_dispatch_instrumented_records = sum(
        not record.get("_diagnostics_missing", False)
        and record.get("action_metrics_version") == 2
        for record in records
    )
    action_dispatch_metrics_available = (
        action_dispatch_instrumented_records == len(records)
    )

    # Keep the raw AgentData classification above for quarantine/audit semantics,
    # while building a strict user-facing real/fake partition for the tracker.
    # A tracker-real row either has a valid Docker result or an explicit preset
    # reward.  Everything else is tracker-fake, including diagnostics missing.
    tracker_counts: Counter[str] = Counter()
    tracker_counts["total"] = len(records)

    def record_tracker_reward(record: dict[str, Any]) -> None:
        reward = record.get("reward_score")
        if reward is None or not math.isfinite(reward):
            tracker_counts["real_reward_other"] += 1
        elif reward == 1.0:
            tracker_counts["real_reward_1"] += 1
        elif reward == 0.0:
            tracker_counts["real_reward_0"] += 1
        elif reward < 0.0:
            tracker_counts["real_reward_negative"] += 1
        else:
            tracker_counts["real_reward_other"] += 1

    def record_tracker_set_reason(outcome: str) -> None:
        if outcome == "assistant_turn_limit":
            tracker_counts["set_terminated_turns"] += 1
        elif outcome == "response_length_limit":
            tracker_counts["set_terminated_response"] += 1
        elif outcome == "agent_terminated_other":
            tracker_counts["set_terminated_other"] += 1
        elif outcome == "patch_empty":
            tracker_counts["set_patch_empty"] += 1
        elif outcome in SET_APPLY_FAILURE_OUTCOMES:
            tracker_counts["set_patch_apply_failed"] += 1
        elif outcome == "action_timeout_limit":
            tracker_counts["set_agent_timeout"] += 1
        elif outcome == "repeated_action_limit":
            tracker_counts["set_repeated_action"] += 1
        else:
            tracker_counts["set_other"] += 1

    def record_tracker_fake_reason(record: dict[str, Any], outcome: str) -> None:
        if outcome in FAKE_TASK_SKIPPED_OUTCOMES:
            tracker_counts["fake_task_skipped"] += 1
        elif outcome in FAKE_GLOBAL_TIMEOUT_OUTCOMES:
            tracker_counts["fake_rollout_global_timeout"] += 1
        elif outcome in FAKE_ROLLOUT_ENV_OUTCOMES or outcome == "reset_git_log_failed":
            tracker_counts["fake_rollout_env_failed"] += 1
        elif outcome in FAKE_EVAL_ENV_OUTCOMES:
            tracker_counts["fake_eval_env_failed"] += 1
        elif outcome in FAKE_EVAL_TIMEOUT_OUTCOMES:
            tracker_counts["fake_eval_timeout"] += 1
        elif outcome in FAKE_EVAL_ERROR_OUTCOMES:
            tracker_counts["fake_eval_execution_error"] += 1
        elif str(record.get("final_stage", "unknown")) in {
            "eval_env_init",
            "apply_patch",
            "evaluation",
            "tests",
        }:
            tracker_counts["fake_eval_other"] += 1
        else:
            tracker_counts["fake_rollout_other"] += 1

    for record in records:
        if record.get("_diagnostics_missing", False):
            counts["diagnostics_missing"] += 1
            tracker_counts["diagnostics_missing"] += 1
            tracker_counts["fake_data"] += 1
            tracker_counts["fake_diagnostics_missing"] += 1
            continue

        is_fake = bool(record.get("is_fake", False))
        did_real_eval = bool(record.get("did_real_eval", False))
        outcome = str(record.get("terminal_outcome", "unknown"))
        if (
            (is_fake and did_real_eval)
            or (did_real_eval and outcome not in REAL_EVAL_OUTCOMES)
            or (not did_real_eval and outcome in REAL_EVAL_OUTCOMES)
        ):
            conflicts += 1

        if is_fake:
            tracker_counts["raw_agent_data_fake"] += 1

        is_tracker_docker_eval = (
            not is_fake
            and did_real_eval
            and outcome in REAL_EVAL_OUTCOMES
        )
        is_tracker_set_eval = (
            not is_fake
            and not did_real_eval
            and outcome in TRACKER_SET_EVAL_OUTCOMES
        )

        if is_tracker_docker_eval:
            tracker_counts["real_data"] += 1
            tracker_counts["real_eval"] += 1
            record_tracker_reward(record)
        elif is_tracker_set_eval:
            tracker_counts["real_data"] += 1
            tracker_counts["set_eval"] += 1
            record_tracker_set_reason(outcome)
        else:
            tracker_counts["fake_data"] += 1
            if not is_fake and outcome in NONFAKE_INVALID_EVAL_OUTCOMES:
                tracker_counts["reclassified_eval_failure"] += 1
            elif not is_fake:
                tracker_counts["reclassified_other_failure"] += 1
            record_tracker_fake_reason(record, outcome)

        if is_fake:
            counts["fake_data"] += 1
            if outcome in FAKE_TASK_SKIPPED_OUTCOMES:
                counts["fake_task_skipped"] += 1
            elif outcome in FAKE_GLOBAL_TIMEOUT_OUTCOMES:
                counts["fake_rollout_global_timeout"] += 1
            elif outcome in FAKE_ROLLOUT_ENV_OUTCOMES:
                counts["fake_rollout_env_failed"] += 1
            elif outcome in FAKE_EVAL_ENV_OUTCOMES:
                counts["fake_eval_env_failed"] += 1
            elif outcome in FAKE_EVAL_TIMEOUT_OUTCOMES:
                counts["fake_eval_timeout"] += 1
            elif outcome in FAKE_EVAL_ERROR_OUTCOMES:
                counts["fake_eval_execution_error"] += 1
            else:
                if str(record.get("final_stage", "unknown")) in {
                    "eval_env_init",
                    "apply_patch",
                    "evaluation",
                    "tests",
                }:
                    counts["fake_eval_other"] += 1
                else:
                    counts["fake_rollout_other"] += 1
            continue

        counts["real_data"] += 1
        if did_real_eval:
            counts["real_eval"] += 1
            reward = record.get("reward_score")
            if reward is None or not math.isfinite(reward):
                counts["real_reward_other"] += 1
            elif reward == 1.0:
                counts["real_reward_1"] += 1
            elif reward == 0.0:
                counts["real_reward_0"] += 1
            elif reward < 0.0:
                counts["real_reward_negative"] += 1
            else:
                counts["real_reward_other"] += 1
        elif outcome in NONFAKE_INVALID_EVAL_OUTCOMES:
            # These rows keep the real response tokens but have no valid test
            # result.  They are neither a successful Docker eval nor fake data.
            counts["invalid_eval_nonfake"] += 1
        else:
            counts["set_eval"] += 1
            if outcome == "assistant_turn_limit":
                counts["set_terminated_turns"] += 1
            elif outcome == "response_length_limit":
                counts["set_terminated_response"] += 1
            elif outcome == "agent_terminated_other":
                counts["set_terminated_other"] += 1
            elif outcome == "patch_empty":
                counts["set_patch_empty"] += 1
            elif outcome in SET_APPLY_FAILURE_OUTCOMES:
                counts["set_patch_apply_failed"] += 1
            elif outcome == "action_timeout_limit":
                counts["set_agent_timeout"] += 1
            elif outcome == "repeated_action_limit":
                counts["set_repeated_action"] += 1
            else:
                counts["set_other"] += 1

    counts["set_agent_terminated"] = (
        counts["set_terminated_turns"]
        + counts["set_terminated_response"]
        + counts["set_terminated_other"]
    )
    counts["fake_eval_failed"] = (
        counts["fake_eval_env_failed"]
        + counts["fake_eval_timeout"]
        + counts["fake_eval_execution_error"]
        + counts["fake_eval_other"]
    )
    counts["fake_other"] = (
        counts["fake_rollout_other"] + counts["fake_eval_other"]
    )
    tracker_counts["set_agent_terminated"] = (
        tracker_counts["set_terminated_turns"]
        + tracker_counts["set_terminated_response"]
        + tracker_counts["set_terminated_other"]
    )
    tracker_counts["fake_eval_failed"] = (
        tracker_counts["fake_eval_env_failed"]
        + tracker_counts["fake_eval_timeout"]
        + tracker_counts["fake_eval_execution_error"]
        + tracker_counts["fake_eval_other"]
    )
    tracker_counts["fake_other"] = (
        tracker_counts["fake_rollout_other"]
        + tracker_counts["fake_eval_other"]
    )

    total = counts["total"]
    real_eval_children = (
        counts["real_reward_1"]
        + counts["real_reward_0"]
        + counts["real_reward_negative"]
        + counts["real_reward_other"]
    )
    set_children = (
        counts["set_agent_terminated"]
        + counts["set_patch_empty"]
        + counts["set_patch_apply_failed"]
        + counts["set_agent_timeout"]
        + counts["set_repeated_action"]
        + counts["set_other"]
    )
    fake_children = (
        counts["fake_task_skipped"]
        + counts["fake_rollout_global_timeout"]
        + counts["fake_rollout_env_failed"]
        + counts["fake_rollout_other"]
        + counts["fake_eval_failed"]
    )
    if total != counts["real_data"] + counts["fake_data"] + counts["diagnostics_missing"]:
        raise AssertionError("core real/fake/missing counts must equal observed trajectories")
    if counts["real_data"] != counts["real_eval"] + counts["set_eval"] + counts["invalid_eval_nonfake"]:
        raise AssertionError("core non-fake subtypes must equal real-data count")
    if counts["real_eval"] != real_eval_children:
        raise AssertionError("real-eval reward buckets must equal real-eval count")
    if counts["set_eval"] != set_children:
        raise AssertionError("set-eval buckets must equal set-eval count")
    if counts["fake_data"] != fake_children:
        raise AssertionError("fake-data buckets must equal fake-data count")

    tracker_real_eval_children = (
        tracker_counts["real_reward_1"]
        + tracker_counts["real_reward_0"]
        + tracker_counts["real_reward_negative"]
        + tracker_counts["real_reward_other"]
    )
    tracker_set_children = (
        tracker_counts["set_agent_terminated"]
        + tracker_counts["set_patch_empty"]
        + tracker_counts["set_patch_apply_failed"]
        + tracker_counts["set_agent_timeout"]
        + tracker_counts["set_repeated_action"]
        + tracker_counts["set_other"]
    )
    tracker_fake_children = (
        tracker_counts["fake_task_skipped"]
        + tracker_counts["fake_rollout_global_timeout"]
        + tracker_counts["fake_rollout_env_failed"]
        + tracker_counts["fake_rollout_other"]
        + tracker_counts["fake_eval_failed"]
        + tracker_counts["fake_diagnostics_missing"]
    )
    if total != tracker_counts["real_data"] + tracker_counts["fake_data"]:
        raise AssertionError("tracker real/fake counts must equal observed trajectories")
    if tracker_counts["real_data"] != tracker_counts["real_eval"] + tracker_counts["set_eval"]:
        raise AssertionError("tracker real count must equal Docker eval plus set eval")
    if tracker_counts["real_eval"] != tracker_real_eval_children:
        raise AssertionError("tracker Docker reward buckets must equal Docker eval count")
    if tracker_counts["set_eval"] != tracker_set_children:
        raise AssertionError("tracker set-eval buckets must equal set-eval count")
    if tracker_counts["fake_data"] != tracker_fake_children:
        raise AssertionError("tracker fake-reason buckets must equal tracker fake count")
    tracker_fake_provenance = (
        tracker_counts["raw_agent_data_fake"]
        + tracker_counts["reclassified_eval_failure"]
        + tracker_counts["reclassified_other_failure"]
        + tracker_counts["fake_diagnostics_missing"]
    )
    if tracker_counts["fake_data"] != tracker_fake_provenance:
        raise AssertionError("tracker fake provenance must equal tracker fake count")

    action_total = int(event_occurrences.get("action_attempt", 0))
    action_parsed = int(event_occurrences.get("action_parse_success", 0))
    action_dispatched = int(event_occurrences.get("action_execute_attempt", 0))
    action_executed = int(event_occurrences.get("action_execute_success", 0))
    if not action_dispatch_metrics_available:
        action_dispatched = action_parsed
    if (
        action_parsed > action_total
        or action_dispatched > action_parsed
        or action_executed > action_dispatched
    ):
        raise AssertionError(
            "action counts must satisfy executed <= dispatched <= parsed <= total"
        )

    trajectory_keys = (
        "total",
        "real_data",
        "real_eval",
        "real_reward_1",
        "real_reward_0",
        "real_reward_negative",
        "real_reward_other",
        "set_eval",
        "set_agent_terminated",
        "set_terminated_turns",
        "set_terminated_response",
        "set_terminated_other",
        "set_patch_empty",
        "set_patch_apply_failed",
        "set_agent_timeout",
        "set_repeated_action",
        "set_other",
        "invalid_eval_nonfake",
        "fake_data",
        "fake_task_skipped",
        "fake_rollout_global_timeout",
        "fake_rollout_env_failed",
        "fake_rollout_other",
        "fake_eval_failed",
        "fake_eval_env_failed",
        "fake_eval_timeout",
        "fake_eval_execution_error",
        "fake_eval_other",
        "fake_other",
        "diagnostics_missing",
    )
    trajectory_counts = {key: int(counts.get(key, 0)) for key in trajectory_keys}
    tracker_trajectory_keys = (
        "total",
        "real_data",
        "real_eval",
        "real_reward_1",
        "real_reward_0",
        "real_reward_negative",
        "real_reward_other",
        "set_eval",
        "set_agent_terminated",
        "set_terminated_turns",
        "set_terminated_response",
        "set_terminated_other",
        "set_patch_empty",
        "set_patch_apply_failed",
        "set_agent_timeout",
        "set_repeated_action",
        "set_other",
        "fake_data",
        "fake_task_skipped",
        "fake_rollout_global_timeout",
        "fake_rollout_env_failed",
        "fake_rollout_other",
        "fake_eval_failed",
        "fake_eval_env_failed",
        "fake_eval_timeout",
        "fake_eval_execution_error",
        "fake_eval_other",
        "fake_diagnostics_missing",
        "fake_other",
        "diagnostics_missing",
        "reclassified_eval_failure",
        "reclassified_other_failure",
        "raw_agent_data_fake",
    )
    tracker_trajectory_counts = {
        key: int(tracker_counts.get(key, 0)) for key in tracker_trajectory_keys
    }
    return {
        "trajectory_counts": trajectory_counts,
        "trajectory_rates": {
            key: _rate(count, total) for key, count in trajectory_counts.items()
        },
        "tracker_trajectory_counts": tracker_trajectory_counts,
        "tracker_trajectory_rates": {
            key: _rate(count, total)
            for key, count in tracker_trajectory_counts.items()
        },
        "action_counts": {
            "total": action_total,
            "parsed_success": action_parsed,
            "dispatched": action_dispatched,
            "executed_success": action_executed,
        },
        "action_rates": {
            "parsed_success": _rate(action_parsed, action_total),
            "dispatched_of_parsed": _rate(action_dispatched, action_parsed),
            "executed_success": _rate(action_executed, action_dispatched),
        },
        "availability": {
            "reward_metrics": bool(core_metrics_available),
            "action_metrics": bool(core_metrics_available),
            "action_dispatch_metrics": bool(action_dispatch_metrics_available),
            "instrumented_records": int(instrumented_records),
        },
        "audit": {
            "expected_trajectories": expected_trajectories,
            "missing_outputs": int(missing_outputs),
            "excess_outputs": int(excess_outputs),
            "unexpected_outputs": int(unexpected_outputs),
            "conflicting_flags": int(conflicts),
        },
    }


def aggregate_step_diagnostics(
    records: Iterable[Any],
    group_uids: Iterable[Any],
    *,
    phase: str,
    step: int,
    expected_group_size: int,
    expected_group_uids: Iterable[Any] | None = None,
    max_examples_per_outcome: int = 3,
) -> dict[str, Any]:
    """Aggregate trajectory diagnostics into a deterministic step summary."""

    normalized = [_normalized_record(record) for record in records]
    uids = [str(uid) for uid in group_uids]
    if len(normalized) != len(uids):
        raise ValueError("records and group_uids must have the same length")

    expected_uids: list[str] | None = None
    expected_trajectories: int | None = None
    missing_outputs = 0
    excess_outputs = 0
    unexpected_outputs = 0
    # Keep diagnostics observational: a session with no TQ output is an audit
    # gap, not a fabricated real/fake trajectory in the observed denominator.
    if expected_group_uids is not None and expected_group_size > 0:
        observed = Counter(uids)
        expected_uids = list(dict.fromkeys(str(item) for item in expected_group_uids))
        expected_trajectories = len(expected_uids) * expected_group_size
        missing_outputs = sum(
            max(expected_group_size - observed.get(uid, 0), 0) for uid in expected_uids
        )
        excess_outputs = sum(
            max(observed.get(uid, 0) - expected_group_size, 0) for uid in expected_uids
        )
        expected_uid_set = set(expected_uids)
        unexpected_outputs = sum(
            count for uid, count in observed.items() if uid not in expected_uid_set
        )

    outcomes = Counter(record["terminal_outcome"] for record in normalized)
    owners = Counter(str(record.get("failure_owner", "unknown")) for record in normalized)
    stages = Counter(str(record.get("final_stage", "unknown")) for record in normalized)
    event_occurrences: Counter[str] = Counter()
    event_trajectories: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_counts_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for uid, record in zip(uids, normalized, strict=True):
        groups[uid].append(record)
        for event, count in record.get("events", {}).items():
            event_occurrences[event] += int(count)
            event_trajectories[event] += 1
        outcome = record["terminal_outcome"]
        task_counts_by_outcome[outcome][str(record.get("task_id", "unknown"))] += 1
        if len(examples[outcome]) < max_examples_per_outcome:
            examples[outcome].append(
                {
                    "task_id": record.get("task_id", "unknown"),
                    "request_id": record.get("request_id", "unknown"),
                    "detail": str(record.get("failure_detail", ""))[:300],
                }
            )

    group_health = Counter()
    partial_fake_rows = 0
    partial_real_rows = 0
    group_ids = expected_uids if expected_uids is not None else list(groups)
    for uid in group_ids:
        rows = groups.get(uid, [])
        if expected_group_size > 0 and len(rows) != expected_group_size:
            group_health["incomplete"] += 1
            continue
        if any(bool(row.get("_diagnostics_missing", False)) for row in rows):
            group_health["diagnostics_missing"] += 1
            continue
        fake_rows = sum(bool(row.get("is_fake", False)) for row in rows)
        if fake_rows == 0:
            group_health["all_real"] += 1
        elif fake_rows == len(rows):
            group_health["all_fake"] += 1
        else:
            group_health["partial_fake"] += 1
            partial_fake_rows += fake_rows
            partial_real_rows += len(rows) - fake_rows

    total = len(normalized)
    fake_count = sum(bool(record.get("is_fake", False)) for record in normalized)
    real_eval_count = sum(bool(record.get("did_real_eval", False)) for record in normalized)
    invalid_eval_count = sum(bool(record.get("invalid_eval", False)) for record in normalized)
    infra_failure_count = sum(bool(record.get("infra_failure", False)) for record in normalized)
    missing_count = sum(bool(record.get("_diagnostics_missing", False)) for record in normalized)
    core = _build_core_summary(
        normalized,
        event_occurrences,
        expected_trajectories=expected_trajectories,
        missing_outputs=missing_outputs,
        excess_outputs=excess_outputs,
        unexpected_outputs=unexpected_outputs,
    )
    timing = _build_timing_summary(
        normalized,
        action_dispatched=core["action_counts"]["dispatched"],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "tracker_layout_version": 6,
        "phase": phase,
        "step": int(step),
        "trajectories": total,
        "outcomes": {name: int(outcomes.get(name, 0)) for name in TERMINAL_OUTCOMES},
        "owners": {name: int(owners.get(name, 0)) for name in OWNER_NAMES},
        "stages": {name: int(stages.get(name, 0)) for name in STAGE_NAMES},
        "events": {
            name: {
                "occurrences": int(event_occurrences.get(name, 0)),
                "trajectories": int(event_trajectories.get(name, 0)),
            }
            for name in EVENT_NAMES
        },
        "fake": {"count": fake_count, "rate": fake_count / total if total else 0.0},
        "tracker_classification": {
            "real": {
                "count": core["tracker_trajectory_counts"]["real_data"],
                "rate": core["tracker_trajectory_rates"]["real_data"],
            },
            "fake": {
                "count": core["tracker_trajectory_counts"]["fake_data"],
                "rate": core["tracker_trajectory_rates"]["fake_data"],
            },
            "raw_agent_data_fake": {
                "count": core["tracker_trajectory_counts"]["raw_agent_data_fake"],
                "rate": core["tracker_trajectory_rates"]["raw_agent_data_fake"],
            },
            "reclassified_eval_failure": {
                "count": core["tracker_trajectory_counts"][
                    "reclassified_eval_failure"
                ],
                "rate": core["tracker_trajectory_rates"][
                    "reclassified_eval_failure"
                ],
            },
            "reclassified_other_failure": {
                "count": core["tracker_trajectory_counts"][
                    "reclassified_other_failure"
                ],
                "rate": core["tracker_trajectory_rates"][
                    "reclassified_other_failure"
                ],
            },
        },
        "real_eval": {
            "count": real_eval_count,
            "rate": real_eval_count / total if total else 0.0,
        },
        "invalid_eval": {
            "count": invalid_eval_count,
            "rate": invalid_eval_count / total if total else 0.0,
        },
        "infra_failure": {
            "count": infra_failure_count,
            "rate": infra_failure_count / total if total else 0.0,
        },
        "diagnostics_missing": missing_count,
        "groups": {
            "total": len(group_ids),
            "expected_size": int(expected_group_size),
            "all_real": int(group_health.get("all_real", 0)),
            "all_fake": int(group_health.get("all_fake", 0)),
            "partial_fake": int(group_health.get("partial_fake", 0)),
            "incomplete": int(group_health.get("incomplete", 0)),
            "diagnostics_missing": int(group_health.get("diagnostics_missing", 0)),
            "partial_fake_rows": int(partial_fake_rows),
            "partial_real_rows_affected": int(partial_real_rows),
        },
        "examples": {name: values for name, values in examples.items() if values},
        "top_tasks": {
            outcome: [
                {"task_id": task_id, "count": int(count)}
                for task_id, count in counts.most_common(max_examples_per_outcome)
            ]
            for outcome, counts in task_counts_by_outcome.items()
            if counts
        },
        "core": core,
        "timing": timing,
    }
    if sum(summary["outcomes"].values()) != total:
        raise AssertionError("terminal outcome counts must equal trajectory count")
    return summary


def flatten_step_diagnostics(summary: dict[str, Any]) -> dict[str, float]:
    """Export five phase-specific SwanLab sections; rich detail stays in JSON."""

    phase = str(summary["phase"])
    is_train = phase == "train"
    core = summary["core"]
    counts = core["tracker_trajectory_counts"]
    availability = core["availability"]
    metrics: dict[str, float] = {}

    # 01: One observed-trajectory count plus compact rates over that observed
    # denominator.  real/fake is a strict tracker partition.  real includes only
    # valid Docker evals and trajectories assigned a preset score.  Evaluation
    # failures and diagnostics-missing rows are tracker-fake; the raw AgentData
    # is_fake marker remains separate in the rich JSON for quarantine auditing.
    overview = "swe1train" if is_train else "swe1val"
    observed_total = counts["total"]
    eval_failures = counts["fake_eval_failed"] + counts["set_patch_apply_failed"]
    metrics.update(
        {
            f"{overview}/traj_cnt": float(observed_total),
            f"{overview}/real_rate": _rate(
                counts["real_data"], observed_total
            ),
            f"{overview}/docker_eval_rate": _rate(
                counts["real_eval"], observed_total
            ),
            f"{overview}/set_eval_rate": _rate(
                counts["set_eval"], observed_total
            ),
            f"{overview}/fake_rate": _rate(
                counts["fake_data"], observed_total
            ),
            f"{overview}/eval_fail_rate": _rate(
                eval_failures, observed_total
            ),
            f"{overview}/miss_rate": _rate(
                counts["diagnostics_missing"], observed_total
            ),
        }
    )

    # 02/03: Use all observed trajectories as the denominator so every
    # trajectory-level outcome can be compared directly across the dashboard.
    # Reward composition remains conditional on a valid Docker evaluation.
    real_root = "swe2trainreal" if is_train else "swe3valreal"
    docker_total = counts["real_eval"]
    metrics.update(
        {
            f"{real_root}/docker_eval_rate": _rate(
                docker_total, observed_total
            ),
            f"{real_root}/set_eval_rate": _rate(
                counts["set_eval"], observed_total
            ),
            f"{real_root}/resp_limit_rate": _rate(
                counts["set_terminated_response"], observed_total
            ),
            f"{real_root}/turn_limit_rate": _rate(
                counts["set_terminated_turns"], observed_total
            ),
            f"{real_root}/act_timeout_rate": _rate(
                counts["set_agent_timeout"], observed_total
            ),
            f"{real_root}/repeat_act_rate": _rate(
                counts["set_repeated_action"], observed_total
            ),
            f"{real_root}/patch_empty_rate": _rate(
                counts["set_patch_empty"], observed_total
            ),
            f"{real_root}/patch_fail_rate": _rate(
                counts["set_patch_apply_failed"], observed_total
            ),
        }
    )
    if availability["reward_metrics"]:
        for leaf, source in (
            ("reward1_rate", "real_reward_1"),
            ("reward0_rate", "real_reward_0"),
            ("rewardneg_rate", "real_reward_negative"),
            ("rewardother_rate", "real_reward_other"),
        ):
            metrics[f"{real_root}/{leaf}"] = _rate(counts[source], docker_total)
    metrics[f"{real_root}/set_other_rate"] = _rate(
        counts["set_terminated_other"] + counts["set_other"],
        observed_total,
    )

    # 04/05: Fake count plus mutually-exclusive fake fallback reasons.  The
    # reason rates use all observed trajectories, so their sum equals the
    # overview fake_rate rather than one.
    fake_root = "swe4trainfake" if is_train else "swe5valfake"
    fake_total = counts["fake_data"]
    metrics[f"{fake_root}/fake_cnt"] = float(fake_total)
    for leaf, source in (
        ("rollout_env_fail_rate", "fake_rollout_env_failed"),
        ("rollout_timeout_rate", "fake_rollout_global_timeout"),
        ("eval_env_fail_rate", "fake_eval_env_failed"),
        ("eval_timeout_rate", "fake_eval_timeout"),
        ("eval_error_rate", "fake_eval_execution_error"),
        ("eval_other_rate", "fake_eval_other"),
        ("task_skip_rate", "fake_task_skipped"),
        ("rollout_other_error_rate", "fake_rollout_other"),
        ("diag_miss_rate", "fake_diagnostics_missing"),
    ):
        metrics[f"{fake_root}/{leaf}"] = _rate(counts[source], observed_total)

    # 06: per-trajectory wall time.  Mean answers the capacity question while
    # p95 makes CPU/contention tails visible without a large chart explosion.
    timing = summary.get("timing", {})
    timing_fields = timing.get("fields", {})
    timing_root = "swe6traintime" if is_train else "swe6valtime"
    for display, source in (
        ("traj_s", "trajectory_wall_seconds"),
        ("rollout_env_init_s", "rollout_env_init_seconds"),
        ("llm_s", "llm_seconds"),
        ("env_step_s", "env_step_seconds"),
        ("eval_s", "final_eval_seconds"),
    ):
        stats = timing_fields.get(source, {})
        for statistic in ("mean", "p95"):
            value = stats.get(statistic)
            if value is not None:
                metrics[f"{timing_root}/{display}_{statistic}"] = float(value)

    # 07: Action funnel plus the pooled caller-observed ENV RPC wall time.
    # Parse failures are excluded from dispatched Action latency by design.
    if availability["action_metrics"] and availability["action_dispatch_metrics"]:
        actions = core["action_counts"]
        action_rates = core["action_rates"]
        action_root = "swe7trainact" if is_train else "swe7valact"
        metrics.update(
            {
                f"{action_root}/act_cnt": float(actions["total"]),
                f"{action_root}/parse_success_rate": float(
                    action_rates["parsed_success"]
                ),
                f"{action_root}/dispatch_cnt": float(
                    actions["dispatched"]
                ),
                f"{action_root}/execute_success_rate": float(
                    action_rates["executed_success"]
                ),
            }
        )
        action_timing_mean = timing.get("action", {}).get("mean_seconds")
        if action_timing_mean is not None:
            metrics[f"{action_root}/env_step_s_mean"] = float(action_timing_mean)
    return metrics


def format_step_diagnostics(summary: dict[str, Any]) -> str:
    """Render the first-phase flat report with one trajectory denominator."""

    core = summary["core"]
    count = core["tracker_trajectory_counts"]
    rate = core["tracker_trajectory_rates"]

    def item(name: str) -> str:
        return f"{count[name]} ({rate[name]:.2%})"

    limits = summary.get("limits", {})
    action_timeout_label = "Agent超时"
    if limits.get("action_timeout_limit"):
        action_timeout_label += f"（累计{limits['action_timeout_limit']}次Action超时）"
    eval_timeout_label = "执行超时"
    if limits.get("eval_timeout_seconds"):
        eval_timeout_label += f"（>{limits['eval_timeout_seconds']}秒）"

    reward_detail = ""
    if core["availability"]["reward_metrics"]:
        reward_detail = (
            " [1分=" + item("real_reward_1")
            + "，0分=" + item("real_reward_0")
            + "，负分=" + item("real_reward_negative") + "]"
        )
    else:
        reward_detail = " [分数统计不可用：旧记录没有reward埋点]"

    lines = [
        f"[SWE-DIAG][{summary['phase']}][step={summary['step']}] "
        f"总轨迹={count['total']}；以下轨迹占比均以总轨迹为分母",
        "  real轨迹=" + item("real_data")
        + "；Docker真实评估=" + item("real_eval")
        + reward_detail,
        "  set评估=" + item("set_eval")
        + " [Agent截断=" + item("set_agent_terminated")
        + "：轮次=" + item("set_terminated_turns")
        + "，上下文=" + item("set_terminated_response")
        + "；Patch空=" + item("set_patch_empty")
        + "；Patch错误=" + item("set_patch_apply_failed")
        + f"；{action_timeout_label}=" + item("set_agent_timeout")
        + "；重复Action熔断=" + item("set_repeated_action") + "]",
        "  fake轨迹=" + item("fake_data")
        + " [任务跳过=" + item("fake_task_skipped")
        + "；轨迹全局超时=" + item("fake_rollout_global_timeout")
        + "；交互环境启动失败=" + item("fake_rollout_env_failed")
        + "；评估失败=" + item("fake_eval_failed")
        + "：评估环境启动失败=" + item("fake_eval_env_failed")
        + f"，{eval_timeout_label}=" + item("fake_eval_timeout")
        + "，执行错误=" + item("fake_eval_execution_error")
        + "；诊断缺失=" + item("fake_diagnostics_missing") + "]",
    ]
    for name, label in (
        ("real_reward_other", "其他真实评估分数"),
        ("set_terminated_other", "其他Agent截断"),
        ("set_other", "其他set评估"),
        ("fake_other", "其他fake"),
    ):
        if name == "real_reward_other" and not core["availability"]["reward_metrics"]:
            continue
        if count[name]:
            lines.append(f"  补充：{label}={item(name)}")

    if (
        core["availability"]["action_metrics"]
        and core["availability"]["action_dispatch_metrics"]
    ):
        actions = core["action_counts"]
        action_rates = core["action_rates"]
        lines.append(
            f"  Action：总数={actions['total']}；成功解析={actions['parsed_success']} "
            f"({action_rates['parsed_success']:.2%})；实际下发={actions['dispatched']}；"
            f"成功执行={actions['executed_success']} "
            f"({action_rates['executed_success']:.2%}，以实际下发为分母)"
        )
    else:
        lines.append("  Action：统计不可用（旧记录没有Action漏斗埋点）")
    audit = core["audit"]
    if any(
        audit[name]
        for name in ("missing_outputs", "excess_outputs", "unexpected_outputs", "conflicting_flags")
    ):
        lines.append(
            f"  完整性告警：缺失输出={audit['missing_outputs']}，"
            f"超额输出={audit['excess_outputs']}，意外UID输出={audit['unexpected_outputs']}，"
            f"冲突标记={audit['conflicting_flags']}"
        )
    return "\n".join(lines)


def write_step_diagnostics(summary: dict[str, Any], base_dir: str) -> str:
    """Atomically overwrite one phase/step JSON file (resume-safe)."""

    phase_dir = os.path.join(base_dir, str(summary["phase"]))
    os.makedirs(phase_dir, exist_ok=True)
    final_path = os.path.join(phase_dir, f"{int(summary['step'])}.json")
    temp_path = f"{final_path}.tmp-{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, final_path)
    return final_path
