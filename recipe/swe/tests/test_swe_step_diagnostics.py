import json
import math
from copy import deepcopy

from recipe.swe.step_diagnostics import (
    aggregate_step_diagnostics,
    finalize_trajectory_diagnostics,
    flatten_step_diagnostics,
    format_step_diagnostics,
    new_trajectory_diagnostics,
    record_action_timing,
    record_event,
    record_timing,
    select_final_session_indices,
    set_terminal_outcome,
    write_step_diagnostics,
)


def _finalize(diag, report=None, *, fake=False, task_id="task-1", reward=0.0):
    return finalize_trajectory_diagnostics(
        diag,
        report=report or {},
        agent_state="finished",
        did_real_eval=bool(report),
        is_fake=fake,
        request_id=f"request-{task_id}",
        task_id=task_id,
        patch_length=42,
        reward_score=reward,
    )


def _assert_tracker_fake_provenance(counts):
    assert counts["fake_data"] == (
        counts["raw_agent_data_fake"]
        + counts["reclassified_eval_failure"]
        + counts["reclassified_other_failure"]
        + counts["fake_diagnostics_missing"]
    )


def test_disabled_has_no_record():
    assert new_trajectory_diagnostics(False, "train") is None


def test_events_do_not_override_successful_terminal_outcome():
    diag = new_trajectory_diagnostics(True, "train")
    record_event(diag, "action_parse_failed", 2)
    record_event(diag, "submission_protocol_rejected")
    result = _finalize(
        diag,
        {"resolved": True, "f2p_rate": 1.0, "p2p_rate": 1.0, "msg": "evaluated"},
    )
    assert result["terminal_outcome"] == "resolved_full"
    assert result["events"] == {
        "action_parse_failed": 2,
        "submission_protocol_rejected": 1,
    }


def test_specific_timeout_outcome_wins_over_finalizer():
    diag = new_trajectory_diagnostics(True, "train")
    record_event(diag, "action_timeout", 3)
    set_terminal_outcome(
        diag,
        "action_timeout_limit",
        stage="action",
        owner="model",
        detail="3/3",
    )
    result = _finalize(diag)
    assert result["terminal_outcome"] == "action_timeout_limit"
    assert result["events"]["action_timeout"] == 3


def test_harness_failure_is_structured_without_inventing_fake_tokens():
    diag = new_trajectory_diagnostics(True, "validation")
    result = _finalize(
        diag,
        {
            "eval_failure_code": "tests_timeout",
            "msg": "tests timed out",
            "f2p_rate": 0,
            "p2p_rate": 0,
        },
        fake=False,
    )
    assert result["terminal_outcome"] == "tests_timeout"
    assert result["is_fake"] is False
    assert result["invalid_eval"] is True
    assert result["did_real_eval"] is False


def test_eval_report_missing_is_preserved_without_inventing_fake_tokens():
    result = _finalize(
        new_trajectory_diagnostics(True, "validation"),
        {"eval_failure_code": "eval_report_missing", "msg": "no report"},
        fake=False,
    )
    assert result["terminal_outcome"] == "eval_report_missing"
    assert result["is_fake"] is False
    assert result["invalid_eval"] is True


def test_unresolved_test_subtypes_are_mutually_exclusive():
    partial = _finalize(
        new_trajectory_diagnostics(True, "train"),
        {"f2p_rate": 0.5, "p2p_rate": 1.0},
    )
    regression = _finalize(
        new_trajectory_diagnostics(True, "train"),
        {"f2p_rate": 1.0, "p2p_rate": 0.75},
    )
    no_progress = _finalize(
        new_trajectory_diagnostics(True, "train"),
        {"f2p_rate": 0.0, "p2p_rate": 1.0},
    )
    assert partial["terminal_outcome"] == "resolved_partial"
    assert regression["terminal_outcome"] == "unresolved_regression"
    assert no_progress["terminal_outcome"] == "unresolved_no_progress"


def test_select_final_session_indices_excludes_padding_and_intermediate_outputs():
    keys = ["uidA_0_0", "uidA_0_1", "uidA_1_0", "uidB_0_0", "uidB_0_1"]
    padding = [False, False, False, False, True]
    assert select_final_session_indices(keys, padding) == [1, 2, 3]


def test_aggregate_reports_partial_fake_and_event_occurrence_counts():
    records = []
    uids = []
    for index in range(8):
        diag = new_trajectory_diagnostics(True, "train")
        if index < 7:
            set_terminal_outcome(
                diag,
                "eval_timeout",
                stage="evaluation",
                owner="evaluator",
                detail="500s",
            )
            fake = True
        else:
            set_terminal_outcome(
                diag,
                "assistant_turn_limit",
                stage="rollout",
                owner="model",
                detail="200 turns",
            )
            fake = False
        record_event(diag, "action_parse_failed", index)
        records.append(_finalize(diag, fake=fake, task_id="reproman-544"))
        uids.append("prompt-1")

    summary = aggregate_step_diagnostics(
        records,
        uids,
        phase="train",
        step=40,
        expected_group_size=8,
    )
    assert sum(summary["outcomes"].values()) == 8
    assert summary["outcomes"]["eval_timeout"] == 7
    assert summary["outcomes"]["assistant_turn_limit"] == 1
    assert summary["events"]["action_parse_failed"] == {
        "occurrences": sum(range(8)),
        "trajectories": 7,
    }
    assert summary["groups"]["partial_fake"] == 1
    assert summary["groups"]["partial_fake_rows"] == 7
    assert summary["groups"]["partial_real_rows_affected"] == 1


def test_core_summary_uses_one_trajectory_denominator_and_action_denominator():
    records = []
    uids = []

    for reward, report in (
        (1.0, {"resolved": True, "f2p_rate": 1.0, "p2p_rate": 1.0}),
        (0.0, {"f2p_rate": 0.0, "p2p_rate": 1.0}),
        (-0.2, {"f2p_rate": 0.0, "p2p_rate": 1.0}),
    ):
        diag = new_trajectory_diagnostics(True, "validation")
        record_event(diag, "action_attempt", 4)
        record_event(diag, "action_parse_success", 3)
        record_event(diag, "action_execute_attempt", 2)
        record_event(diag, "action_execute_success", 2)
        records.append(_finalize(diag, report, reward=reward))
        uids.append(f"real-{len(uids)}")

    for outcome in (
        "assistant_turn_limit",
        "response_length_limit",
        "patch_empty",
        "patch_apply_failed",
        "action_timeout_limit",
    ):
        diag = new_trajectory_diagnostics(True, "validation")
        set_terminal_outcome(diag, outcome, stage="rollout", owner="model")
        records.append(_finalize(diag, reward=-0.2))
        uids.append(f"set-{len(uids)}")

    invalid = new_trajectory_diagnostics(True, "validation")
    records.append(
        _finalize(
            invalid,
            {"eval_failure_code": "test_parser_empty"},
            reward=0.0,
        )
    )
    uids.append("invalid")

    for outcome in (
        "prompt_too_long",
        "rollout_env_init_failed",
        "eval_env_init_failed",
        "eval_timeout",
        "eval_execution_failed",
    ):
        diag = new_trajectory_diagnostics(True, "validation")
        set_terminal_outcome(diag, outcome, stage="evaluation", owner="infra")
        records.append(_finalize(diag, fake=True))
        uids.append(f"fake-{len(uids)}")

    summary = aggregate_step_diagnostics(
        records,
        uids,
        phase="validation",
        step=0,
        expected_group_size=1,
    )
    core = summary["core"]
    counts = core["trajectory_counts"]
    tracker_counts = core["tracker_trajectory_counts"]
    assert counts["total"] == 14
    assert counts["real_data"] == 9
    assert counts["real_eval"] == 3
    assert counts["real_reward_1"] == 1
    assert counts["real_reward_0"] == 1
    assert counts["real_reward_negative"] == 1
    assert counts["set_eval"] == 5
    assert counts["set_patch_apply_failed"] == 1
    assert counts["invalid_eval_nonfake"] == 1
    assert counts["fake_data"] == 5
    assert counts["fake_eval_failed"] == 3
    assert core["trajectory_rates"]["real_eval"] == 3 / 14
    assert tracker_counts["total"] == 14
    assert tracker_counts["real_data"] == 8
    assert tracker_counts["real_eval"] == 3
    assert tracker_counts["set_eval"] == 5
    assert tracker_counts["fake_data"] == 6
    assert tracker_counts["raw_agent_data_fake"] == 5
    assert tracker_counts["reclassified_eval_failure"] == 1
    assert tracker_counts["reclassified_other_failure"] == 0
    assert tracker_counts["diagnostics_missing"] == 0
    _assert_tracker_fake_provenance(tracker_counts)
    assert summary["tracker_classification"]["real"]["count"] == 8
    assert summary["tracker_classification"]["fake"]["count"] == 6
    assert core["action_counts"] == {
        "total": 12,
        "parsed_success": 9,
        "dispatched": 6,
        "executed_success": 6,
    }
    assert core["action_rates"] == {
        "parsed_success": 0.75,
        "dispatched_of_parsed": 2 / 3,
        "executed_success": 1.0,
    }
    metrics = flatten_step_diagnostics(summary)
    assert metrics["swe1val/traj_cnt"] == 14.0
    assert metrics["swe1val/real_rate"] == 8 / 14
    assert metrics["swe1val/docker_eval_rate"] == 3 / 14
    assert metrics["swe1val/set_eval_rate"] == 5 / 14
    assert metrics["swe1val/fake_rate"] == 6 / 14
    assert metrics["swe1val/eval_fail_rate"] == (1 + 3 + 1) / 14
    assert metrics["swe1val/miss_rate"] == 0.0

    assert metrics["swe3valreal/docker_eval_rate"] == 3 / 14
    assert metrics["swe3valreal/set_eval_rate"] == 5 / 14
    assert "swe3valreal/trajreal_but_evalfail_rate" not in metrics
    expected_set_rates = {
        "resp_limit_rate": 1 / 14,
        "turn_limit_rate": 1 / 14,
        "act_timeout_rate": 1 / 14,
        "repeat_act_rate": 0.0,
        "patch_empty_rate": 1 / 14,
        "patch_fail_rate": 1 / 14,
        "set_other_rate": 0.0,
    }
    for leaf, expected in expected_set_rates.items():
        assert metrics[f"swe3valreal/{leaf}"] == expected
    assert math.isclose(
        sum(expected_set_rates.values()), metrics["swe3valreal/set_eval_rate"]
    )
    assert metrics["swe3valreal/reward1_rate"] == 1 / 3
    assert metrics["swe3valreal/reward0_rate"] == 1 / 3
    assert metrics["swe3valreal/rewardneg_rate"] == 1 / 3
    assert metrics["swe3valreal/rewardother_rate"] == 0.0
    assert math.isclose(
        metrics["swe3valreal/docker_eval_rate"]
        + metrics["swe3valreal/set_eval_rate"],
        metrics["swe1val/real_rate"],
    )
    assert math.isclose(
        metrics["swe1val/real_rate"] + metrics["swe1val/fake_rate"],
        1.0,
    )

    expected_fake_rates = {
        "rollout_env_fail_rate": 1 / 14,
        "rollout_timeout_rate": 0.0,
        "eval_env_fail_rate": 1 / 14,
        "eval_timeout_rate": 1 / 14,
        "eval_error_rate": 2 / 14,
        "eval_other_rate": 0.0,
        "task_skip_rate": 1 / 14,
        "rollout_other_error_rate": 0.0,
        "diag_miss_rate": 0.0,
    }
    assert metrics["swe5valfake/fake_cnt"] == 6.0
    for leaf, expected in expected_fake_rates.items():
        assert metrics[f"swe5valfake/{leaf}"] == expected
    assert math.isclose(
        sum(expected_fake_rates.values()), metrics["swe1val/fake_rate"]
    )
    assert math.isclose(
        metrics["swe5valfake/fake_cnt"] / metrics["swe1val/traj_cnt"],
        metrics["swe1val/fake_rate"],
    )
    assert math.isclose(
        metrics["swe3valreal/patch_fail_rate"]
        + metrics["swe5valfake/eval_env_fail_rate"]
        + metrics["swe5valfake/eval_timeout_rate"]
        + metrics["swe5valfake/eval_error_rate"]
        + metrics["swe5valfake/eval_other_rate"],
        metrics["swe1val/eval_fail_rate"],
    )

    summary["limits"] = {"action_timeout_limit": 3, "eval_timeout_seconds": 500}
    rendered = format_step_diagnostics(summary)
    assert "累计3次Action超时" in rendered
    assert "执行超时（>500秒）" in rendered
    assert "以实际下发为分母" in rendered


def test_nonfake_invalid_eval_is_tracker_fake_without_mutating_raw_marker():
    record = _finalize(
        new_trajectory_diagnostics(True, "validation"),
        {"eval_failure_code": "test_parser_empty"},
        fake=False,
        reward=0.0,
    )
    assert record["is_fake"] is False
    original_record = deepcopy(record)
    summary = aggregate_step_diagnostics(
        [record],
        ["uid"],
        phase="validation",
        step=0,
        expected_group_size=1,
    )
    counts = summary["core"]["trajectory_counts"]
    assert counts["real_data"] == 1
    assert counts["real_eval"] == 0
    assert counts["set_eval"] == 0
    assert counts["invalid_eval_nonfake"] == 1
    assert counts["fake_data"] == 0
    tracker_counts = summary["core"]["tracker_trajectory_counts"]
    assert tracker_counts["real_data"] == 0
    assert tracker_counts["fake_data"] == 1
    assert tracker_counts["reclassified_eval_failure"] == 1
    assert tracker_counts["reclassified_other_failure"] == 0
    assert tracker_counts["raw_agent_data_fake"] == 0
    assert tracker_counts["fake_eval_execution_error"] == 1
    _assert_tracker_fake_provenance(tracker_counts)
    assert summary["fake"]["count"] == 0
    assert summary["groups"]["all_real"] == 1
    metrics = flatten_step_diagnostics(summary)
    assert metrics["swe1val/real_rate"] == 0.0
    assert metrics["swe1val/fake_rate"] == 1.0
    assert metrics["swe5valfake/fake_cnt"] == 1.0
    assert metrics["swe5valfake/eval_error_rate"] == 1.0
    assert "swe3valreal/trajreal_but_evalfail_rate" not in metrics
    assert record == original_record
    assert record["is_fake"] is False


def test_nonfake_nonset_outcomes_are_tracker_fake_with_specific_reasons():
    cases = (
        (
            "unknown",
            "unknown",
            "unknown",
            "fake_rollout_other",
            "rollout_other_error_rate",
        ),
        (
            "prompt_too_long",
            "prompt",
            "model",
            "fake_task_skipped",
            "task_skip_rate",
        ),
        (
            "rollout_env_init_failed",
            "rollout_env_init",
            "infra",
            "fake_rollout_env_failed",
            "rollout_env_fail_rate",
        ),
        (
            "reset_git_log_failed",
            "rollout_env_init",
            "infra",
            "fake_rollout_env_failed",
            "rollout_env_fail_rate",
        ),
        (
            "rollout_global_timeout",
            "rollout",
            "model",
            "fake_rollout_global_timeout",
            "rollout_timeout_rate",
        ),
        (
            "rollout_internal_error",
            "rollout",
            "infra",
            "fake_rollout_other",
            "rollout_other_error_rate",
        ),
        (
            "rollout_step_infra_failed",
            "rollout",
            "infra",
            "fake_rollout_other",
            "rollout_other_error_rate",
        ),
        (
            "trajectory_output_missing",
            "rollout",
            "infra",
            "fake_rollout_other",
            "rollout_other_error_rate",
        ),
        (
            "skipped_other",
            "prompt",
            "infra",
            "fake_task_skipped",
            "task_skip_rate",
        ),
        (
            "resolved_full",
            "tests",
            "model",
            "fake_eval_other",
            "eval_other_rate",
        ),
    )
    records = []
    for outcome, stage, owner, _, _ in cases:
        diag = new_trajectory_diagnostics(True, "train")
        if outcome != "unknown":
            set_terminal_outcome(diag, outcome, stage=stage, owner=owner)
        records.append(_finalize(diag, fake=False))

    original_records = deepcopy(records)
    assert all(record["is_fake"] is False for record in records)
    assert all(record["did_real_eval"] is False for record in records)
    for index, ((_, _, _, expected_reason, expected_leaf), record) in enumerate(
        zip(cases, records, strict=True)
    ):
        single_summary = aggregate_step_diagnostics(
            [record],
            [f"single-{index}"],
            phase="train",
            step=1,
            expected_group_size=1,
        )
        single_counts = single_summary["core"]["tracker_trajectory_counts"]
        assert single_counts["real_data"] == 0
        assert single_counts["set_eval"] == 0
        assert single_counts["fake_data"] == 1
        assert single_counts["reclassified_other_failure"] == 1
        assert single_counts[expected_reason] == 1
        _assert_tracker_fake_provenance(single_counts)
        single_metrics = flatten_step_diagnostics(single_summary)
        assert single_metrics[f"swe4trainfake/{expected_leaf}"] == 1.0

    summary = aggregate_step_diagnostics(
        records,
        [f"uid-{index}" for index in range(len(records))],
        phase="train",
        step=1,
        expected_group_size=1,
    )

    raw_counts = summary["core"]["trajectory_counts"]
    assert raw_counts["real_data"] == 10
    assert raw_counts["set_eval"] == 10
    assert raw_counts["set_other"] == 10
    assert raw_counts["fake_data"] == 0
    assert summary["fake"]["count"] == 0
    assert summary["groups"]["all_real"] == 10

    tracker_counts = summary["core"]["tracker_trajectory_counts"]
    assert tracker_counts["real_data"] == 0
    assert tracker_counts["real_eval"] == 0
    assert tracker_counts["set_eval"] == 0
    assert tracker_counts["fake_data"] == 10
    assert tracker_counts["raw_agent_data_fake"] == 0
    assert tracker_counts["reclassified_eval_failure"] == 0
    assert tracker_counts["reclassified_other_failure"] == 10
    assert tracker_counts["fake_task_skipped"] == 2
    assert tracker_counts["fake_rollout_env_failed"] == 2
    assert tracker_counts["fake_rollout_global_timeout"] == 1
    assert tracker_counts["fake_rollout_other"] == 4
    assert tracker_counts["fake_eval_other"] == 1
    _assert_tracker_fake_provenance(tracker_counts)
    assert (
        summary["tracker_classification"]["reclassified_other_failure"]["count"]
        == 10
    )

    metrics = flatten_step_diagnostics(summary)
    assert metrics["swe1train/real_rate"] == 0.0
    assert metrics["swe1train/docker_eval_rate"] == 0.0
    assert metrics["swe1train/set_eval_rate"] == 0.0
    assert metrics["swe1train/fake_rate"] == 1.0
    assert metrics["swe1train/eval_fail_rate"] == 1 / 10
    assert metrics["swe4trainfake/fake_cnt"] == 10.0
    assert metrics["swe4trainfake/task_skip_rate"] == 2 / 10
    assert metrics["swe4trainfake/rollout_env_fail_rate"] == 2 / 10
    assert metrics["swe4trainfake/rollout_timeout_rate"] == 1 / 10
    assert metrics["swe4trainfake/rollout_other_error_rate"] == 4 / 10
    assert metrics["swe4trainfake/eval_other_rate"] == 1 / 10
    assert summary["core"]["audit"]["conflicting_flags"] == 1
    assert records == original_records
    assert all(record["is_fake"] is False for record in records)


def test_raw_reset_git_log_reason_stays_other_but_tracker_maps_env_failure():
    diag = new_trajectory_diagnostics(True, "train")
    set_terminal_outcome(
        diag,
        "reset_git_log_failed",
        stage="rollout_env_init",
        owner="infra",
    )
    record = _finalize(diag, fake=True)
    original_record = deepcopy(record)

    summary = aggregate_step_diagnostics(
        [record],
        ["uid"],
        phase="train",
        step=1,
        expected_group_size=1,
    )
    raw_counts = summary["core"]["trajectory_counts"]
    assert raw_counts["fake_data"] == 1
    assert raw_counts["fake_rollout_env_failed"] == 0
    assert raw_counts["fake_rollout_other"] == 1
    assert summary["fake"]["count"] == 1
    assert summary["groups"]["all_fake"] == 1

    tracker_counts = summary["core"]["tracker_trajectory_counts"]
    assert tracker_counts["fake_data"] == 1
    assert tracker_counts["raw_agent_data_fake"] == 1
    assert tracker_counts["reclassified_eval_failure"] == 0
    assert tracker_counts["reclassified_other_failure"] == 0
    assert tracker_counts["fake_rollout_env_failed"] == 1
    assert tracker_counts["fake_rollout_other"] == 0
    _assert_tracker_fake_provenance(tracker_counts)

    metrics = flatten_step_diagnostics(summary)
    assert metrics["swe4trainfake/fake_cnt"] == 1.0
    assert metrics["swe4trainfake/rollout_env_fail_rate"] == 1.0
    assert metrics["swe4trainfake/rollout_other_error_rate"] == 0.0
    assert record == original_record
    assert record["is_fake"] is True
    assert record["terminal_outcome"] == "reset_git_log_failed"


def test_all_nonfake_invalid_eval_outcomes_move_to_tracker_fake_reasons():
    outcomes = (
        "eval_env_init_timeout",
        "eval_env_init_failed",
        "eval_timeout",
        "eval_output_limit",
        "eval_execution_failed",
        "harness_reset_failed",
        "tests_timeout",
        "tests_error",
        "test_markers_missing",
        "test_parser_empty",
        "eval_report_missing",
    )
    records = []
    for outcome in outcomes:
        diag = new_trajectory_diagnostics(True, "validation")
        set_terminal_outcome(
            diag,
            outcome,
            stage="eval_env_init" if outcome.startswith("eval_env_init") else "evaluation",
            owner="evaluator",
        )
        records.append(_finalize(diag, fake=False))

    summary = aggregate_step_diagnostics(
        [*records, None],
        [*(f"uid-{index}" for index in range(len(records))), "missing"],
        phase="validation",
        step=0,
        expected_group_size=1,
    )
    raw_counts = summary["core"]["trajectory_counts"]
    tracker_counts = summary["core"]["tracker_trajectory_counts"]
    assert raw_counts["invalid_eval_nonfake"] == 11
    assert raw_counts["fake_data"] == 0
    assert raw_counts["diagnostics_missing"] == 1
    assert tracker_counts["real_data"] == 0
    assert tracker_counts["fake_data"] == 12
    assert tracker_counts["reclassified_eval_failure"] == 11
    assert tracker_counts["reclassified_other_failure"] == 0
    assert tracker_counts["fake_eval_env_failed"] == 2
    assert tracker_counts["fake_eval_timeout"] == 2
    assert tracker_counts["fake_eval_execution_error"] == 7
    assert tracker_counts["fake_diagnostics_missing"] == 1
    _assert_tracker_fake_provenance(tracker_counts)

    metrics = flatten_step_diagnostics(summary)
    assert metrics["swe1val/real_rate"] == 0.0
    assert metrics["swe1val/fake_rate"] == 1.0
    assert metrics["swe1val/eval_fail_rate"] == 11 / 12
    assert metrics["swe5valfake/fake_cnt"] == 12.0
    assert metrics["swe5valfake/eval_env_fail_rate"] == 2 / 12
    assert metrics["swe5valfake/eval_timeout_rate"] == 2 / 12
    assert metrics["swe5valfake/eval_error_rate"] == 7 / 12
    assert metrics["swe5valfake/diag_miss_rate"] == 1 / 12


def test_old_records_do_not_report_missing_reward_or_action_counters_as_zero():
    record = _finalize(
        new_trajectory_diagnostics(True, "validation"),
        {"resolved": True, "f2p_rate": 1.0, "p2p_rate": 1.0},
        reward=1.0,
    )
    record.pop("core_metrics_version")
    record.pop("action_metrics_version")
    record.pop("timing_metrics_version")
    record.pop("timing")
    record.pop("reward_score")
    summary = aggregate_step_diagnostics(
        [record],
        ["uid"],
        phase="validation",
        step=0,
        expected_group_size=1,
    )
    assert summary["core"]["availability"]["reward_metrics"] is False
    assert summary["core"]["availability"]["action_metrics"] is False
    rendered = format_step_diagnostics(summary)
    assert "分数统计不可用" in rendered
    assert "Action：统计不可用" in rendered
    metrics = flatten_step_diagnostics(summary)
    reward_leaves = {
        "reward1_rate",
        "reward0_rate",
        "rewardneg_rate",
        "rewardother_rate",
    }
    assert not any(key.rsplit("/", 1)[-1] in reward_leaves for key in metrics)
    assert not any(key.startswith("swe7valact/") for key in metrics)
    assert not any(key.startswith("swe6valtime/") for key in metrics)
    assert not any(key.startswith("swe_diag/") for key in metrics)
    assert summary["timing"]["fields"]["trajectory_wall_seconds"]["mean"] is None


def test_core_tracker_keys_are_stable_for_zero_and_nonzero_requested_buckets():
    empty = aggregate_step_diagnostics(
        [], [], phase="train", step=1, expected_group_size=8
    )
    diag = new_trajectory_diagnostics(True, "train")
    record_event(diag, "action_attempt")
    record_event(diag, "action_parse_success")
    record_event(diag, "action_execute_attempt")
    record_event(diag, "action_execute_success")
    set_terminal_outcome(diag, "eval_timeout", stage="evaluation", owner="infra")
    nonzero = aggregate_step_diagnostics(
        [_finalize(diag, fake=True)],
        ["uid"],
        phase="train",
        step=2,
        expected_group_size=1,
    )
    empty_metrics = flatten_step_diagnostics(empty)
    assert set(empty_metrics) == set(flatten_step_diagnostics(nonzero))
    assert empty_metrics["swe1train/traj_cnt"] == 0.0
    assert empty_metrics["swe4trainfake/fake_cnt"] == 0.0
    assert all(
        value == 0.0
        for key, value in empty_metrics.items()
        if key.endswith("_rate")
    )
    assert "swe7trainact/env_step_s_mean" not in empty_metrics


def test_bad_report_rates_and_nonfinite_rewards_do_not_break_diagnostics():
    for reward in (None, float("nan"), float("inf"), 0.5):
        record = _finalize(
            new_trajectory_diagnostics(True, "validation"),
            {"f2p_rate": "bad", "p2p_rate": object()},
            reward=reward,
        )
        summary = aggregate_step_diagnostics(
            [record],
            ["uid"],
            phase="validation",
            step=0,
            expected_group_size=1,
        )
        assert summary["core"]["trajectory_counts"]["real_reward_other"] == 1


def test_action_funnel_guard_rejects_impossible_counts():
    diag = new_trajectory_diagnostics(True, "train")
    record_event(diag, "action_attempt", 1)
    record_event(diag, "action_parse_success", 2)
    try:
        aggregate_step_diagnostics(
            [_finalize(diag)],
            ["uid"],
            phase="train",
            step=1,
            expected_group_size=1,
        )
    except AssertionError as exc:
        assert "executed <= dispatched <= parsed <= total" in str(exc)
    else:
        raise AssertionError("impossible Action counts must fail the diagnostic guard")


def test_missing_record_keeps_denominator_and_fixed_tracker_keys():
    summary = aggregate_step_diagnostics(
        [None],
        ["prompt-1"],
        phase="validation",
        step=0,
        expected_group_size=1,
    )
    assert summary["trajectories"] == 1
    assert summary["outcomes"]["unknown"] == 1
    assert summary["diagnostics_missing"] == 1
    assert summary["groups"]["diagnostics_missing"] == 1
    assert summary["groups"]["all_real"] == 0
    assert summary["fake"]["count"] == 0
    tracker_counts = summary["core"]["tracker_trajectory_counts"]
    assert tracker_counts["real_data"] == 0
    assert tracker_counts["fake_data"] == 1
    assert tracker_counts["fake_diagnostics_missing"] == 1
    assert tracker_counts["raw_agent_data_fake"] == 0
    assert tracker_counts["reclassified_eval_failure"] == 0
    assert tracker_counts["reclassified_other_failure"] == 0
    _assert_tracker_fake_provenance(tracker_counts)
    metrics = flatten_step_diagnostics(summary)
    assert summary["phase"] == "validation"
    assert metrics["swe1val/traj_cnt"] == 1.0
    assert metrics["swe1val/miss_rate"] == 1.0
    for leaf in (
        "real_rate",
        "docker_eval_rate",
        "set_eval_rate",
        "eval_fail_rate",
    ):
        assert metrics[f"swe1val/{leaf}"] == 0.0
    assert metrics["swe1val/fake_rate"] == 1.0
    assert metrics["swe5valfake/fake_cnt"] == 1.0
    assert metrics["swe5valfake/diag_miss_rate"] == 1.0
    assert all(key.count("/") == 1 for key in metrics)
    assert not any("/outcome/" in key for key in metrics)


def test_expected_groups_account_for_sessions_without_tq_output():
    diag = _finalize(
        new_trajectory_diagnostics(True, "train"),
        {"resolved": True, "f2p_rate": 1.0, "p2p_rate": 1.0},
    )
    summary = aggregate_step_diagnostics(
        [diag],
        ["uid-a"],
        phase="train",
        step=1,
        expected_group_size=2,
        expected_group_uids=["uid-a", "uid-b"],
    )
    assert summary["trajectories"] == 1
    assert summary["outcomes"]["trajectory_output_missing"] == 0
    assert summary["diagnostics_missing"] == 0
    assert summary["groups"]["incomplete"] == 2
    assert summary["core"]["audit"] == {
        "expected_trajectories": 4,
        "missing_outputs": 3,
        "excess_outputs": 0,
        "unexpected_outputs": 0,
        "conflicting_flags": 0,
    }


def test_expected_output_audit_separates_missing_excess_and_unexpected_uids():
    record = _finalize(new_trajectory_diagnostics(True, "train"))
    summary = aggregate_step_diagnostics(
        [record, record, record],
        ["uid-a", "uid-a", "uid-x"],
        phase="train",
        step=1,
        expected_group_size=1,
        expected_group_uids=["uid-a", "uid-b"],
    )
    audit = summary["core"]["audit"]
    assert summary["trajectories"] == 3
    assert audit["expected_trajectories"] == 2
    assert audit["missing_outputs"] == 1
    assert audit["excess_outputs"] == 1
    assert audit["unexpected_outputs"] == 1


def test_valid_unknown_is_not_missing_and_unknown_events_are_bounded():
    diag = new_trajectory_diagnostics(True, "train")
    record_event(diag, "arbitrary-user-controlled-value")
    summary = aggregate_step_diagnostics(
        [diag],
        ["prompt-1"],
        phase="train",
        step=1,
        expected_group_size=1,
    )
    assert summary["outcomes"]["unknown"] == 1
    assert summary["diagnostics_missing"] == 0
    assert summary["events"]["unknown_event"]["occurrences"] == 1
    assert "arbitrary-user-controlled-value" not in summary["events"]


def test_repeated_action_limit_is_a_distinct_set_eval_outcome():
    diag = new_trajectory_diagnostics(True, "validation")
    record_event(diag, "repeated_action", 2)
    set_terminal_outcome(
        diag,
        "repeated_action_limit",
        stage="action",
        owner="model",
        detail="3 identical commands",
    )
    summary = aggregate_step_diagnostics(
        [_finalize(diag, reward=0.0)],
        ["uid"],
        phase="validation",
        step=0,
        expected_group_size=1,
    )
    counts = summary["core"]["trajectory_counts"]
    assert counts["real_data"] == 1
    assert counts["set_eval"] == 1
    assert counts["set_repeated_action"] == 1
    assert summary["events"]["repeated_action"] == {
        "occurrences": 2,
        "trajectories": 1,
    }
    assert flatten_step_diagnostics(summary)["swe3valreal/repeat_act_rate"] == 1.0


def test_rollout_global_timeout_has_a_dedicated_fake_bucket():
    diag = new_trajectory_diagnostics(True, "validation")
    set_terminal_outcome(
        diag,
        "rollout_global_timeout",
        stage="rollout",
        owner="model",
        detail="3600s",
    )
    summary = aggregate_step_diagnostics(
        [_finalize(diag, fake=True)],
        ["uid"],
        phase="validation",
        step=0,
        expected_group_size=1,
    )
    counts = summary["core"]["trajectory_counts"]
    assert counts["fake_data"] == 1
    assert counts["fake_rollout_global_timeout"] == 1
    metrics = flatten_step_diagnostics(summary)
    assert metrics["swe5valfake/fake_cnt"] == 1.0
    assert metrics["swe5valfake/rollout_timeout_rate"] == 1.0
    assert "轨迹全局超时=1 (100.00%)" in format_step_diagnostics(summary)


def test_json_write_is_resume_safe_atomic_overwrite(tmp_path):
    summary = aggregate_step_diagnostics(
        [],
        [],
        phase="train",
        step=3,
        expected_group_size=8,
    )
    path = write_step_diagnostics(summary, str(tmp_path))
    path2 = write_step_diagnostics(summary, str(tmp_path))
    assert path2 == path
    parsed = json.loads((tmp_path / "train" / "3.json").read_text())
    assert parsed["trajectories"] == 0
    assert not list((tmp_path / "train").glob("*.tmp-*"))


def _timed_summary(phase):
    diag = new_trajectory_diagnostics(True, phase)
    record_event(diag, "action_attempt", 2)
    record_event(diag, "action_parse_success", 2)
    record_event(diag, "action_execute_attempt", 2)
    record_event(diag, "action_execute_success")
    record_action_timing(diag, 1.0)
    record_action_timing(diag, 3.0)
    for name, value in (
        ("trajectory_wall_seconds", 10.0),
        ("rollout_env_init_seconds", 1.0),
        ("interaction_seconds", 7.0),
        ("llm_seconds", 2.0),
        ("final_eval_seconds", 2.0),
    ):
        record_timing(diag, name, value)
    record = _finalize(
        diag,
        {"resolved": True, "f2p_rate": 1.0, "p2p_rate": 1.0},
        reward=1.0,
    )
    return aggregate_step_diagnostics(
        [record],
        ["uid"],
        phase=phase,
        step=1,
        expected_group_size=1,
    )


def test_tracker_uses_exact_v6_schema_order_and_ten_top_level_groups():
    train_summary = _timed_summary("train")
    val_summary = _timed_summary("validation")
    train_metrics = flatten_step_diagnostics(train_summary)
    val_metrics = flatten_step_diagnostics(val_summary)
    metrics = {**train_metrics, **val_metrics}
    assert train_summary["tracker_layout_version"] == 6
    assert val_summary["tracker_layout_version"] == 6

    overview_leaves = [
        "traj_cnt",
        "real_rate",
        "docker_eval_rate",
        "set_eval_rate",
        "fake_rate",
        "eval_fail_rate",
        "miss_rate",
    ]
    real_leaves = [
        "docker_eval_rate",
        "set_eval_rate",
        "resp_limit_rate",
        "turn_limit_rate",
        "act_timeout_rate",
        "repeat_act_rate",
        "patch_empty_rate",
        "patch_fail_rate",
        "reward1_rate",
        "reward0_rate",
        "rewardneg_rate",
        "rewardother_rate",
        "set_other_rate",
    ]
    fake_leaves = [
        "fake_cnt",
        "rollout_env_fail_rate",
        "rollout_timeout_rate",
        "eval_env_fail_rate",
        "eval_timeout_rate",
        "eval_error_rate",
        "eval_other_rate",
        "task_skip_rate",
        "rollout_other_error_rate",
        "diag_miss_rate",
    ]
    timing_leaves = [
        f"{name}_{stat}"
        for name in (
            "traj_s",
            "rollout_env_init_s",
            "llm_s",
            "env_step_s",
            "eval_s",
        )
        for stat in ("mean", "p95")
    ]
    action_leaves = [
        "act_cnt",
        "parse_success_rate",
        "dispatch_cnt",
        "execute_success_rate",
        "env_step_s_mean",
    ]

    expected_train = [
        *(f"swe1train/{leaf}" for leaf in overview_leaves),
        *(f"swe2trainreal/{leaf}" for leaf in real_leaves),
        *(f"swe4trainfake/{leaf}" for leaf in fake_leaves),
        *(f"swe6traintime/{leaf}" for leaf in timing_leaves),
        *(f"swe7trainact/{leaf}" for leaf in action_leaves),
    ]
    expected_val = [
        *(f"swe1val/{leaf}" for leaf in overview_leaves),
        *(f"swe3valreal/{leaf}" for leaf in real_leaves),
        *(f"swe5valfake/{leaf}" for leaf in fake_leaves),
        *(f"swe6valtime/{leaf}" for leaf in timing_leaves),
        *(f"swe7valact/{leaf}" for leaf in action_leaves),
    ]

    assert list(train_metrics) == expected_train
    assert list(val_metrics) == expected_val
    assert len(train_metrics) == 45
    assert len(val_metrics) == 45
    assert set(metrics) == set(expected_train + expected_val)
    assert len(metrics) == 90
    assert {key.split("/", 1)[0] for key in metrics} == {
        "swe1train",
        "swe1val",
        "swe2trainreal",
        "swe3valreal",
        "swe4trainfake",
        "swe5valfake",
        "swe6traintime",
        "swe6valtime",
        "swe7trainact",
        "swe7valact",
    }
    assert all(key.count("/") == 1 for key in metrics)
    assert not any(
        key.startswith(("swe_", "swe_diag/", "swe1/", "swe6time/", "swe7act/"))
        for key in metrics
    )


def test_action_funnel_uses_real_env_dispatch_denominator_and_weighted_time():
    summary = _timed_summary("train")
    actions = summary["core"]["action_counts"]
    rates = summary["core"]["action_rates"]
    assert actions == {
        "total": 2,
        "parsed_success": 2,
        "dispatched": 2,
        "executed_success": 1,
    }
    assert rates["parsed_success"] == 1.0
    assert rates["executed_success"] == 0.5
    assert summary["timing"]["action"]["mean_seconds"] == 2.0
    metrics = flatten_step_diagnostics(summary)
    assert metrics["swe7trainact/act_cnt"] == 2.0
    assert metrics["swe7trainact/parse_success_rate"] == 1.0
    assert metrics["swe7trainact/dispatch_cnt"] == 2.0
    assert metrics["swe7trainact/execute_success_rate"] == 0.5
    assert metrics["swe7trainact/env_step_s_mean"] == 2.0
    assert metrics["swe6traintime/traj_s_mean"] == 10.0
    assert metrics["swe6traintime/traj_s_p95"] == 10.0


def test_timing_aggregation_ignores_missing_values_instead_of_filling_zero():
    timed = new_trajectory_diagnostics(True, "train")
    record_timing(timed, "trajectory_wall_seconds", 10.0)
    legacy = _finalize(new_trajectory_diagnostics(True, "train"))
    legacy.pop("timing_metrics_version")
    legacy.pop("timing")
    summary = aggregate_step_diagnostics(
        [_finalize(timed), legacy],
        ["a", "b"],
        phase="train",
        step=1,
        expected_group_size=1,
    )
    stats = summary["timing"]["fields"]["trajectory_wall_seconds"]
    assert stats == {
        "sample_count": 1,
        "mean": 10.0,
        "p50": 10.0,
        "p95": 10.0,
        "max": 10.0,
    }
    assert summary["timing"]["coverage_rate_of_trajectories"] == 0.5
    metrics = flatten_step_diagnostics(summary)
    assert {
        key for key in metrics if key.startswith("swe6traintime/")
    } == {
        "swe6traintime/traj_s_mean",
        "swe6traintime/traj_s_p95",
    }
