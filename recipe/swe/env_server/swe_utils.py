#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc SWE-utils
@author: plm
@create: 2026-01-30
"""

import hashlib
import re

from jinja2 import StrictUndefined, Template
import yaml
import time

def load_yaml_config_from_file_path(yaml_config_path: str) -> dict:
    with open(yaml_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_swe_eval_format_image_name(instance_id):
    """不适用于swe-rebench-v2"""
    # swe.eval 示例：sweb.eval.x86_64.raullenchai_1776_rapid-mlx-341_interface:latest
    # swe.rebench.v2 示例，在instance里面已经拼好：swerebenchv2_elastic-synthetics_316-f52f0bf:latest
    repo, name = instance_id.split("__", 1)
    image_name = f'sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()
    return image_name


def get_instance_docker_image(
    instance_id: str,
    data_docker_source: str = 'SWE-bench',
    swebench_official_image: bool = True,
    instance=None
) -> str:
    if swebench_official_image:
        normalized_source = data_docker_source.strip().lower().replace('_', '-')
        if normalized_source in {'swe-rebench-v2', 'swe-bench-pro'}:
            assert instance is not None
            image_name = (
                instance.get("docker_image_name") or instance.get("image_name")
            )
            if not image_name:
                raise ValueError(
                    f"Missing prebuilt image name for data source "
                    f"{data_docker_source!r}, instance_id={instance_id!r}"
                )
            return image_name
        if normalized_source == 'swe-bench-live':
            docker_image_prefix = 'docker.io/starryzhang/'
        elif normalized_source == 'swe-bench':
            docker_image_prefix = 'docker.io/swebench/'
        elif normalized_source == 'swe-bench-multilingual':
            docker_image_prefix = 'docker.io/swebench/'
        elif normalized_source in ('swe-rebench', 'swe-rebench-leaderboard'):
            docker_image_prefix = 'docker.io/swerebench/'
        else:
            docker_image_prefix = 'docker.io/swebench/' # default

        repo, name = instance_id.split("__", 1)
        image_name = f'{docker_image_prefix.rstrip("/")}/sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()
        return image_name
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{instance_id.replace('__', '_s_')}".lower()


def resolve_swe_repo_dir(instance: dict) -> str:
    """Resolve the repository root inside an instance image.

    SWE-rebench-V2 images are built with ``WORKDIR /<repo-name>``.  The image
    slug is not a safe source for this value because repository owners may
    themselves contain ``-``.  Other datasets keep their canonical ``repo_dir``
    value (``/testbed`` for the legacy SWE harness and ``/app`` for Pro).
    """

    data_docker_source = (
        str(instance.get("data_docker_source", ""))
        .strip()
        .lower()
        .replace("_", "-")
    )
    if data_docker_source == "swe-rebench-v2":
        repo = str(instance.get("repo", ""))
        owner, separator, repo_name = repo.rpartition("/")
        if not separator or not owner or not repo_name:
            raise ValueError(
                "SWE-rebench-V2 repo must use the 'owner/name' format, "
                f"got {repo!r} for instance_id={instance.get('instance_id')!r}"
            )
        return f"/{repo_name}"

    repo_dir = instance.get("repo_dir")
    if isinstance(repo_dir, str) and repo_dir.strip():
        return repo_dir.strip()
    return "/testbed"



def render_template(template: str, **kwargs) -> str:
    return Template(template, undefined=StrictUndefined).render(
        **kwargs
    )



def get_group_id(instance_id: str, num_groups: int = 2) -> int:
    # 使用 MD5 Hash 确保分配非常均匀且固定
    repo, _ = instance_id.split("__", 1)
    hash_val = int(hashlib.md5(repo.encode()).hexdigest(), 16)
    group_id = hash_val % num_groups
    return group_id




class Timer:
    def __init__(self, can_print=True):
        self.start = self.get_current_time()
        self.last = self.start
        # self.tik()
        self.can_print = can_print
        self.used = 0.0
        self.total = 0.0

    def get_current_time(self):
        return time.monotonic()

    def tik(self):
        self.start = self.get_current_time()
        self.last = self.start
        # print("Timer start at {}".format(self.start))

    def get_print_info_by_seconds(self, seconds):
        return self.get_info(seconds * 1000)

    def get_info(self, ms_count):
        """
        获取准确消耗的时间信息，ms s min h day
        Args:
            ms_count: ms数量
        Returns:
            info: str, n ms/s/min/h/day
        """
        max_ms = 1000 * 5
        max_s = 60 * 5
        max_min = 60 * 2
        max_h = 24 * 3
        ms_count = int(ms_count)
        info = "{}ms".format(ms_count)
        if ms_count >= max_ms:
            # 秒
            s_count = round(ms_count / 1000, 1)
            info = "{}s".format(s_count)
            if s_count >= max_s:
                # 分钟
                min_count = round(s_count/60, 1)
                info = "{}min".format(min_count)
                if min_count >= max_min:
                    # 小时
                    h_count = round(min_count/60, 1)
                    info = "{}h".format(h_count)
                    if h_count >= max_h:
                        # 天
                        day_count = round(h_count/24, 1)
                        info = "{}day".format(day_count)
        return info

    def get_total_used_seconds(self):
        end = self.get_current_time()
        return round(end - self.start, 2)


    def tok(self, info):
        end = self.get_current_time()
        # print("time end: {}".format(end))
        used_seconds = round(end - self.last, 2)
        total_seconds = round(end - self.start, 2)
        # self.used += used
        self.total = total_seconds
        self.last = end
        used_info = self.get_info(used_seconds * 1000)
        total_info = self.get_info(total_seconds * 1000)
        if self.can_print is True:
            print("{}: used={}, total={}".format(info, used_info, total_info))
        return used_seconds, total_seconds, used_info, total_info
