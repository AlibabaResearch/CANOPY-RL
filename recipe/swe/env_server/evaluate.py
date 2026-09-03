#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc evaluate，修改自官方源代码
@author: plm
@create: 2026-02-08
"""
from typing import Any

from swebench.harness.constants import (APPLY_PATCH_FAIL, END_TEST_OUTPUT,
                                        FAIL_ONLY_REPOS, FAIL_TO_FAIL,
                                        FAIL_TO_PASS, KEY_INSTANCE_ID,
                                        KEY_PREDICTION,
                                        MAP_REPO_VERSION_TO_SPECS,
                                        PASS_TO_FAIL, PASS_TO_PASS,
                                        RESET_FAILED, START_TEST_OUTPUT,
                                        TESTS_ERROR, TESTS_TIMEOUT, EvalType,
                                        ResolvedStatus, TestStatus)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, NAME_TO_PARSER
from swebench.harness.test_spec.test_spec import TestSpec


# MARK: Utility functions
def test_passed(case: str, sm: dict[str, str]) -> bool:
    return case in sm and sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]


def test_failed(case: str, sm: dict[str, str]) -> bool:
    return case not in sm or sm[case] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
    ]


# MARK: Evaluation report functions
def get_logs_eval_with_diagnostics(
    test_spec: TestSpec, log_fp: str
) -> tuple[dict[str, str], bool, str]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
    Returns:
        bool: whether the patch applied successfully
        dict: status map

    TODO(john-b-yang): Check this is working properly...
    """
    repo = test_spec.repo
    version = test_spec.version
    # log_parser = MAP_REPO_TO_PARSER[repo]
    # test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if test_spec.install_config is not None:
        # log_parser = MAP_REPO_TO_PARSER[test_spec.install_config['log_parser']]
        log_parser = NAME_TO_PARSER[test_spec.install_config['log_parser']]
        test_cmd = test_spec.install_config['test_cmd']
    else:
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    # print(f"log_parser: {log_parser}, install_config: {test_spec.install_config}")
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    with open(log_fp) as f:
        content = f.read()
        # Preserve the harness subtype for diagnostics while keeping the
        # existing ``found`` behavior and reward semantics unchanged.
        failure_codes = (
            (APPLY_PATCH_FAIL, "harness_apply_patch_failed"),
            (RESET_FAILED, "harness_reset_failed"),
            (TESTS_TIMEOUT, "tests_timeout"),
            (TESTS_ERROR, "tests_error"),
        )
        matched_failure = next((name for marker, name in failure_codes if marker in content), "")
        if matched_failure:
            return {}, False, matched_failure
        elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
            # Test patch did not apply (should not happen at all)
            return {}, False, "test_markers_missing"

        # Get status map of evaluation results
        test_content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]

        # Try parsing the content between markers first
        status_map = log_parser(test_content)

        # If no test results found between markers (common in Modal environment),
        # try parsing the entire log content as fallback
        if not status_map:
            # Look for pytest output patterns in the entire log content
            # This handles cases where pytest output goes to stderr and isn't captured between markers
            status_map = log_parser(content)

        return status_map, True, "" if status_map else "test_parser_empty"


def get_logs_eval(test_spec: TestSpec, log_fp: str) -> tuple[dict[str, str], bool]:
    """Backward-compatible public API; diagnostics use the extended helper."""

    status_map, found, _ = get_logs_eval_with_diagnostics(test_spec, log_fp)
    return status_map, found


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str],
    calculate_to_fail: bool = False,
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_sm (dict): evaluation status map
        gold_results (dict): gold results
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    Metric Definitions (Gold Result Pair + Eval Result):
    - Fail-Pass (F2P) + P: Success (Resolution)
    - Pass-Pass (P2P) + P: Success (Maintenance)
    - Fail-Pass (F2P) + F: Failure
    - Pass-Pass (P2P) + F: Failure

    Miscellaneous Definitions
    - Fail-Fail (F2F) + F: Failure Maintenance
    - Pass-Fail (P2F) + F: Not considered
    - Fail-Fail (F2F) + P: Success (Extra Credit)
    - Pass-Fail (P2F) + P: Not considered
    """

    def check_pass_and_fail(test_case, eval_status_map, success, failed):
        if test_passed(test_case, eval_status_map):
            # Assume silent success for now (test case not in eval_sm)
            success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_fail_only(test_case, eval_status_map, success, failed):
        if (
            test_case in eval_status_map
            and eval_status_map[test_case] == TestStatus.FAILED.value
        ):
            failed.append(test_case)
        else:
            success.append(test_case)

    check_test_case = (
        check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )

    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        check_test_case(test_case, eval_status_map, f2p_success, f2p_failure)

    # Calculate maintenance metrics
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        check_test_case(test_case, eval_status_map, p2p_success, p2p_failure)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
        },
    }

    f2f_success = []
    f2f_failure = []
    p2f_success = []
    p2f_failure = []
    if calculate_to_fail:
        # Calculate "extra credit" metrics
        for test_case in gold_results[FAIL_TO_FAIL]:
            check_test_case(test_case, eval_status_map, f2f_success, f2f_failure)

        # Calculate not considered metrics
        for test_case in gold_results[PASS_TO_FAIL]:
            check_test_case(test_case, eval_status_map, p2f_success, p2f_failure)

    results.update(
        {
            FAIL_TO_FAIL: {
                "success": f2f_success,
                "failure": f2f_failure,
            },
            PASS_TO_FAIL: {
                "success": p2f_success,
                "failure": p2f_failure,
            },
        }
    )
    return results


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute fail-to-pass metric. Accepts single report as argument.
    """
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute pass-to-pass metric. Accepts single report as argument.
    """
    total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
    if total == 0:
        # TODO: Don't factor in p2p metrics
        return 1
    return len(report[PASS_TO_PASS]["success"]) / total


def get_resolution_status(report: dict[str, dict[str, Any]], f2p=None, p2p=None) -> str:
    """
    Determine resolved status of an evaluation instance

    Criteria:
        - If fail-to-pass (Resolution) = 1 and pass-to-pass (Maintenance) = 1 -> FULL
        - If (fail-to-pass (Resolution) < 1 and > 0) and pass-to-pass (Maintenance) = 1 -> PARTIAL
        - Otherwise -> NO
    """
    if f2p is None:
        f2p = compute_fail_to_pass(report)
    if p2p is None:
        p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    else:
        return ResolvedStatus.NO.value


# reward_score = self.compute_reward_score_by_report(report)

def compute_reward_score_by_report(report, use_sparse_reward=True):
    """
    计算最终的reward得分

    设计思路：
    - 完美解决 (f2p=1, p2p=1) → 1.0
    - 部分解决 (f2p>0) → f2p * p2p (f2p决定上限，p2p作为破坏惩罚乘数)
    - 无进展但无破坏 (f2p=0, p2p=1) → 0.05 (微小正向激励，鼓励不搞破坏)
    - 破坏性行为 (f2p=0, p2p<1) → 0.0

    示例：
    - f2p=0.8, p2p=1.0 → 0.8  修了80%的bug，没破坏已有功能
    - f2p=0.8, p2p=0.9 → 0.72 修了80%但破坏了10%已有功能
    - f2p=0.5, p2p=0.5 → 0.25 修了一半毁了一半，方案可能有问题
    - f2p=0.0, p2p=1.0 → 0.05 什么都没改但也没破坏
    """
    resolved = report["resolved"]
    f2p_rate = report["f2p_rate"]
    p2p_rate = report["p2p_rate"]
    if resolved:
        return 1.0
    if use_sparse_reward:
        return 0.0
    if f2p_rate > 0:
        return f2p_rate * p2p_rate
    else:
        if p2p_rate >= 1.0:
            return 0.05
        return 0.0

def get_eval_report(
    test_spec: TestSpec,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
    use_sparse_reward: bool = True
) -> dict[str, Any]:
    """
    Generate a report of model evaluation results from a prediction, task instance,
    and evaluation log.

    Args:
        test_spec (dict): test spec containing keys "instance_id", "FAIL_TO_PASS", and "PASS_TO_PASS"
        prediction (dict): prediction containing keys "instance_id", "model_name_or_path", and "model_patch"
        log_path (str): path to evaluation log
        include_tests_status (bool): whether to include the status of each test in the returned report
    Returns:
        report (dict): report of metrics
    """
    report_map = {}

    instance_id = prediction[KEY_INSTANCE_ID]
    report_map[instance_id] = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "resolved": False,
        "f2p_rate": 0,
        "p2p_rate": 0,
        "resolve_status": ResolvedStatus.NO.value,
        "reward_score": 0,
        "eval_failure_code": "",
    }

    # Check if the model patch exists
    if prediction[KEY_PREDICTION] is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    # Get evaluation logs
    eval_status_map, found, eval_failure_code = get_logs_eval_with_diagnostics(
        test_spec, test_log_path
    )
    report_map[instance_id]["eval_failure_code"] = eval_failure_code

    if not found:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    eval_type = (
        EvalType.FAIL_ONLY
        if test_spec.repo in FAIL_ONLY_REPOS
        else EvalType.PASS_AND_FAIL
    )

    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)


    resolve_status = get_resolution_status(report=report, f2p=f2p, p2p=p2p)
    report_map[instance_id]["resolve_status"] = resolve_status
    report_map[instance_id]["f2p_rate"] = f2p
    report_map[instance_id]["p2p_rate"] = p2p

    if resolve_status == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True

    if include_tests_status:
        report_map[instance_id]["tests_status"] = report  # type: ignore

    model_patch = prediction["model_patch"]
    report_map[instance_id]["reward_score"] = compute_reward_score_by_report(report_map[instance_id], use_sparse_reward=use_sparse_reward)

    return report_map
