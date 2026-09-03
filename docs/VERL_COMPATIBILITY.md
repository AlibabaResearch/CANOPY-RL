# Verl compatibility and porting

CANOPY is a source-only collection of verl recipes and launch commands. The
repository provides one audited source layout and a documented route for
porting the recipe source. Do not confuse portability with a claim of universal
framework compatibility.

## 1. Audited paper-command source layout

The repository bundles the exact Verl Python source snapshot used by the paper
AppWorld command and selected earlier launchers:

```text
commit:   19c6af5de10de2b5272c83c0e82aa715c8c621f3
describe: v0.8.0-11-g19c6af5d
path:     verl/
```

`VERL_ROOT` defaults to the CANOPY repository root, so the launchers use this
snapshot without downloading another framework checkout. The CANOPY delta—nine
content-modified files plus one normalized pathname—is recorded in
`patches/verl-canopy.patch`.

The patch is reproducible only against the recorded upstream commit:

```bash
git -C /path/to/verl checkout --detach 19c6af5de10de2b5272c83c0e82aa715c8c621f3
git -C /path/to/verl apply --check /path/to/CANOPY/patches/verl-canopy.patch
git -C /path/to/verl apply /path/to/CANOPY/patches/verl-canopy.patch
```

## 2. Porting to another verl checkout

The AppWorld and SWE recipe source can be adapted into an operator-managed verl
checkout. A safe starting layout is to place the retained `recipe/` and
`run_scripts/` content in the target checkout, so that both roots refer to that
same tree:

```bash
export CANOPY_ROOT=/path/to/target-verl-with-canopy-recipes
export VERL_ROOT="$CANOPY_ROOT"
```

The earlier AppWorld/SWE launchers were checked with the bundled layout; the
guarded Qwen3.6 0811/0817/0819 configurations additionally require the later
capabilities documented below. Merely setting `VERL_ROOT` to a second checkout
while running from this repository is **not** a supported switch: the bundled
`verl/` package and Ray working directory can otherwise take import precedence
and create a mixed runtime.

Porting is best-effort and unverified until the target commit has been tested.
Do **not** blindly apply the recorded patch to another verl revision. Newer
versions may already provide equivalent behavior, may move the affected
interfaces, or may require a compatibility adaptation. SWE is especially
sensitive because `recipe/swe/main_ppo_sync.py` imports internal verl trainer
APIs. At minimum, verify the following surfaces:

| Surface | Why CANOPY uses it |
| --- | --- |
| custom agent-loop registration | imports `appworld_env_agent` and `swe_agent` before rollout |
| validation-state forwarding | lets a recipe distinguish training from validation |
| sampling stop-token handling | retains the tested Qwen chat termination behavior |
| nested model-config overrides | supports the tested dictionary-backed Megatron configuration |
| CP-aware per-token-loss normalization | prevents DP x CP gradient over-scaling for the Qwen3.6 CP=4 backward pass |
| Megatron router replay | the 0817 MoE command selects R2 routing replay |
| SGLang server interface | follows the rollout backend API available in the tested environment |
| multi-node SGLang port coordination | avoids conflicting NCCL ports across colocated rollout servers |
| rejected native-action credit masking | keeps a rejected tool call in context while removing its generated tokens from policy credit |
| Ray runtime environment | supplies working-directory and multi-node environment settings |

The TransferQueue install-hint change in the patch is diagnostic only; it does
not define framework compatibility. A target revision should pass import and
resolved-config checks, a one-batch Ray smoke test, and an end-to-end recipe
check before it is recorded as compatible. Verify in particular that `verl` is
actually imported from the intended checkout rather than this bundled tree.

The bundled patch includes two later infrastructure refinements used by the
guarded commands: rebuilding the R2 registry against the final live model tree
and reserving a deterministic SGLang NCCL port until server construction. The
retained 0811/0817/0819 commands still require two refinements absent from the
bundled reference: correct DP x CP normalization for the legacy two-value
Megatron loss callback, and keeping a rejected native action in the
conversation while zeroing the policy mask for its generated tokens. The
launchers check the definition **and integration** of all required capabilities
and exit before Ray submission when any is absent. A port may implement
equivalent behavior and adapt the source-marker preflight, but it must then pass
a one-batch multi-node gradient-norm smoke test before the command is described
as runnable or validated.

The bundled SGLang reservation is opt-in through the launcher environment
variables. The standalone evaluation launchers do not enable it because they
are retained earlier command records. Operators running many colocated rollout
replicas may enable the reviewed reservation path after validating their port
range and cluster policy.

## Compatibility claims

- The bundled snapshot plus patch is the audited source-layout reference; this
  is not an end-to-end result-reproduction guarantee.
- Another verl checkout may run a port of the recipes, but no rolling commit is
  currently claimed as verified.
- CANOPY does not install or pin the external checkout's Python/GPU dependency
  graph. The operator must maintain and scan that environment.
- When reporting a new result, record the CANOPY commit, verl commit, resolved
  Hydra configuration, environment/image digest, and dataset/model revisions.

See `recipe/appworld/REQUIRED_VERL.txt` and
`recipe/swe/REQUIRED_VERL.txt` for machine-readable compatibility metadata.
