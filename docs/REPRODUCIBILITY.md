# Reproducibility scope and known boundaries

CANOPY provides source, configuration, and launch commands. A checkout alone
does not reproduce reported numbers; the following external artifacts and
infrastructure are required.

## Shared prerequisites

- The bundled Verl reference documented in `patches/README.md`, or a separately
  validated port to one exact commit under `docs/VERL_COMPATIBILITY.md`.
- An operator-provided Linux/CUDA environment. Historical top-level versions
  in `docs/TESTED_ENVIRONMENT.md` are a record, not an installer or security
  baseline.
- An existing Ray Jobs endpoint and paths accessible to every process that
  consumes models, data, images, or outputs.
- Legally obtained model weights, dataset revisions, benchmark packages, and
  container images. None are distributed here.

For every run, preserve the Git commit, resolved Hydra configuration, Ray job
ID, launcher, model revision, data revision and row count, image digest set,
evaluator revision, output/log digest, and all deviations from the retained
command.

## AppWorld

- Obtain AppWorld through its authorized distribution and comply with its
  protected-content rules.
- The paper prompt is bundled in `recipe/appworld/env_server/prompts.py` and
  pinned by the public-release check. It is the public AppWorld legacy ReAct
  prompt with the final task placeholder removed; the service sends that
  task as a separate user message. `CANOPY_APPWORLD_PROMPT_FILE` optionally
  overrides this default.
- The current service enforces `max_completion_tokens`; the archived experiment
  recipe read this field but did not enforce it. A fresh run is therefore a
  correctness-cleaned reconstruction, not a bitwise replay.
- Generate the four named splits locally and preserve their original names in
  run records.
- The old-environment implementation was not pinned, so no `Old Env` command is
  presented as reproducible evidence.

The retained AppWorld commands and context/sampling differences are listed in
`run_scripts/appworld_readme.md`.

## SWE

### Data and evaluation integrity

- Pin every dataset revision and retain the generated Parquet hash and row
  count. Task instances and source repositories remain subject to their
  upstream terms.
- Generate Parquet only after image archives are available. The builder filters
  missing archives by default; never report that subset as a complete benchmark.
- Before selecting a training partition, exclude every evaluation
  `instance_id` and independently verify a zero train/evaluation intersection.
- SWE-bench Pro needs its authorized per-instance runner/parser files and the
  official timeout/evaluator settings for final reporting.

### Images, cluster, and isolation

- Download images on an authorized host, archive them outside Git, and preload
  only images referenced by the selected Parquet files.
- Advertise `group_0` on the Ray head and one unique `group_<id>` on each
  worker. The Parquet group count, registered resources, and `trainer.nnodes`
  must agree.
- The retained 0817 layout uses 12 nodes. Observed storage was approximately
  3.1 TB per node for more than 16,000 images; this is capacity-planning
  evidence, not a minimum requirement.
- Benchmark repositories execute untrusted code. Use dedicated workers,
  no secrets, restricted egress, firewalled Ray services, and disposable
  node-local storage. Public container networking defaults to `none`, and
  rollout uses preloaded images with `--pull never`.

### Source/runtime boundary

- The 0528, 0621, and 0628 commands were migrated from an archived
  `recipe.swe.main_ppo` name to the retained `recipe.swe.main_ppo_sync` entry
  point. Preserve this migration note when comparing logs.
- `recipe.swe.main_ppo_sync` depends on internal Verl interfaces. Porting it to
  another commit requires compatibility tests rather than a path-only switch.
- The bundled patch includes live R2 registry rebinding and deterministic
  SGLang NCCL-port reservation. The 0811/0817/0819 Qwen3.6 commands still
  require CP-aware per-token-loss normalization and rejected-native-action
  credit masking that are absent from the bundled source. Their launchers fail
  closed and are porting references; do not bypass the guards. Port equivalent
  behavior and pass a one-batch gradient-norm smoke test before a full run.
- The six standalone SWE evaluation launchers evaluate Verified only. The
  guarded 0817 and 0819 training configurations contain joint validation inputs
  for Verified, Multilingual, ReBench leaderboard, and Pro.

## Release traceability

Public archives are built only from a clean Git commit by
`tools/build_public_release.sh`. Record:

1. canonical repository URL and full Git commit;
2. SCA task/result URL and scan time;
3. zero unresolved code-snippet findings;
4. disposition of every license or component finding;
5. archive SHA-256 produced from that same commit.
