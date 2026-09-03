import random
from types import SimpleNamespace

from recipe.swe.image_download import archive_directory, archive_name, resolve_image
from recipe.swe.swe_data_process import (
    _clean_extra_info,
    _read_instance_ids,
    build_records,
)


def test_archive_names_match_retained_layout():
    assert (
        archive_name(
            "docker.io/swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
            "SWE-bench",
        )
        == "swebench_sweb.eval.x86_64.owner_1776_repo-1_latest.tar"
    )
    assert (
        archive_name(
            "docker.io/swerebenchv2/owner.repo-1:latest",
            "SWE-rebench-v2",
        )
        == "docker.io_swerebenchv2_owner.repo-1_latest.tar"
    )
    assert (
        archive_name(
            "jefzda/sweap-images:owner.repo-1",
            "SWE-bench-pro",
        )
        == "jefzda_sweap-images_owner.repo-1.tar"
    )


def test_archive_directory_rejects_image_root_escape(tmp_path):
    try:
        archive_directory(tmp_path, "../escape")
    except ValueError as exc:
        assert "outside --image-root" in str(exc)
    else:
        raise AssertionError("dataset-name path escape was accepted")


def test_clean_removes_all_known_host_paths():
    cleaned = _clean_extra_info(
        {
            "instance_id": "owner__repo-1",
            "local_tar_path": "/host/images/image.tar",
            "local_image_tar_path": "/host/images/image.tar",
            "host_image_path": "/host/images",
            "pro_run_script_path": "/host/pro/run_script.sh",
            "pro_parser_path": "/host/pro/parser.py",
        }
    )

    assert cleaned == {"instance_id": "owner__repo-1"}


def test_instance_id_reader_accepts_newline_file(tmp_path):
    path = tmp_path / "exclude.txt"
    path.write_text("owner__repo-1\nowner__repo-2\n", encoding="utf-8")

    assert _read_instance_ids(str(path)) == {"owner__repo-1", "owner__repo-2"}


def test_part_selection_happens_after_missing_archive_filter(monkeypatch, tmp_path):
    instances = [
        {
            "instance_id": f"owner__repo-{index}",
            "repo": "owner/repo",
            "base_commit": f"commit-{index}",
            "test_patch": "",
            "problem_statement": f"problem {index}",
        }
        for index in range(5)
    ]
    monkeypatch.setattr(
        "recipe.swe.swe_data_process._load_source",
        lambda source, split, revision=None: instances,
    )

    image_root = tmp_path / "images"
    archive_dir = image_root / "verified"
    archive_dir.mkdir(parents=True)
    retained_ids = []
    for instance in instances[:-1]:
        image = resolve_image(instance, "SWE-bench")
        (archive_dir / archive_name(image, "SWE-bench")).write_bytes(b"archive")
        retained_ids.append(instance["instance_id"])

    args = SimpleNamespace(
        source="unused",
        split="test",
        revision=None,
        instance_ids=None,
        exclude_instance_ids=None,
        exclude_repos_from=None,
        exclude_repos_split="test",
        exclude_repos_revision=None,
        part_size=2,
        part_index=1,
        max_count=None,
        seed=42,
        num_groups=2,
        tokenizer=None,
        agent_config=None,
        tokenizer_revision=None,
        max_prompt_length=None,
        image_root=str(image_root),
        dataset_name="verified",
        data_docker_source="SWE-bench",
        keep_missing_images=False,
        swe_bench_pro_run_scripts_dir=None,
    )

    records = build_records(args)
    random.Random(42).shuffle(retained_ids)

    assert [record["extra_info"]["instance_id"] for record in records] == retained_ids[:2]

    # A final partial part must not absorb more than --part-size or disappear
    # because of floor division (four retained rows with part size three).
    tail_args = SimpleNamespace(**vars(args))
    tail_args.part_size = 3
    tail_args.part_index = 2
    tail_records = build_records(tail_args)
    assert [record["extra_info"]["instance_id"] for record in tail_records] == retained_ids[3:]


def test_routing_uses_the_instance_id_repo_key(monkeypatch, tmp_path):
    instances = [
        {
            "instance_id": f"django__django-{index}",
            "repo": "django/django",
            "base_commit": f"commit-{index}",
            "test_patch": "",
            "problem_statement": f"problem {index}",
        }
        for index in range(40)
    ]
    monkeypatch.setattr(
        "recipe.swe.swe_data_process._load_source",
        lambda source, split, revision=None: instances,
    )

    image_root = tmp_path / "images"
    archive_dir = image_root / "verified"
    archive_dir.mkdir(parents=True)
    for instance in instances:
        image = resolve_image(instance, "SWE-bench")
        (archive_dir / archive_name(image, "SWE-bench")).write_bytes(b"archive")

    args = SimpleNamespace(
        source="unused",
        split="test",
        revision=None,
        instance_ids=None,
        exclude_instance_ids=None,
        exclude_repos_from=None,
        exclude_repos_split="test",
        exclude_repos_revision=None,
        part_size=None,
        part_index=1,
        max_count=None,
        seed=42,
        num_groups=4,
        tokenizer=None,
        agent_config=None,
        tokenizer_revision=None,
        max_prompt_length=None,
        image_root=str(image_root),
        dataset_name="verified",
        data_docker_source="SWE-bench",
        keep_missing_images=False,
        swe_bench_pro_run_scripts_dir=None,
    )

    records = build_records(args)
    groups = {record["extra_info"]["group_id"] for record in records}

    assert groups == {0, 1, 2, 3}
