# CANOPY launchers

The retained launchers are grouped by scenario and purpose:

```text
run_scripts/
├── appworld/
│   ├── train/
│   └── eval/
└── swe/
    ├── cluster/
    │   ├── storage.conf
    │   ├── podman.sh
    │   ├── start_head.sh
    │   └── start_worker.sh
    ├── train/
    └── eval/
```

Each launcher contains its own experiment parameters; there is no shared
`common.sh`. Before running a launcher, edit its clearly marked local paths for
the model/checkpoint, generated Parquet files, outputs, and (where applicable)
the Ray dashboard. The commands intentionally do not download models, task
data, benchmark images, or protected prompts.

- [AppWorld guide](appworld_readme.md) · [中文](appworld_readme.zh-CN.md)
- [SWE guide](swe_readme.md) · [中文](swe_readme.zh-CN.md)

The numeric dates and node counts in filenames identify retained experiment
commands, not minimum hardware requirements. Start from the closest launcher
and adjust topology-dependent parameters together.
