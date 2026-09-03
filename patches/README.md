# Tested Verl patch

The repository bundles the tested, modified `verl/` package-source snapshot
used by the paper AppWorld command and selected earlier launchers. The guarded
Qwen3.6 0811/0817/0819 configurations still require additional loss-scaling
and credit-masking capabilities and do not claim to run on this snapshot.
`verl-canopy.patch` is the audit record of
CANOPY's bundled delta and applies only to:

```text
https://github.com/verl-project/verl.git
commit 19c6af5de10de2b5272c83c0e82aa715c8c621f3
describe v0.8.0-11-g19c6af5d
```

It was generated from the CANOPY local repository's retained Verl delta. It is
a provenance and reconstruction record, not a compatibility patch for the
latest Verl branch. When porting a recipe, compare each required behavior with
the target revision; newer code may already contain an equivalent mechanism or
may require a different adaptation. Do not apply this patch blindly.

To verify or reconstruct the bundled snapshot manually:

```bash
git -C /path/to/verl checkout --detach 19c6af5de10de2b5272c83c0e82aa715c8c621f3
git -C /path/to/verl apply --check /path/to/CANOPY/patches/verl-canopy.patch
git -C /path/to/verl apply /path/to/CANOPY/patches/verl-canopy.patch
```

The patch changes:

| Upstream path | CANOPY change |
| --- | --- |
| `verl/experimental/agent_loop/__init__.py` | Register the AppWorld and SWE loops |
| `verl/experimental/agent_loop/agent_loop.py` | Pass validation state and add the chat stop token |
| `verl/experimental/fully_async_policy/shell/grpo_qwen3_235b_megatron_npu.sh<U+200E>` | Normalize the trailing U+200E format character in the upstream filename; content is unchanged |
| `verl/trainer/runtime_env.yaml` | Record the tested multi-node runtime settings |
| `verl/utils/megatron/router_replay_patch.py` | Rebind router replay to the final live MoE model tree |
| `verl/utils/model.py` | Support nested dictionary-backed model config overrides |
| `verl/utils/net_utils.py` | Add an exclusive, bounded TCP port reservation helper |
| `verl/utils/transferqueue_utils.py` | Align the install hint with TransferQueue 0.1.6 |
| `verl/workers/engine/megatron/transformer_impl.py` | Validate and bind the live R2 router registry |
| `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | Match the tested SGLang launch API and reserve single-node NCCL ports |

The patch applies cleanly to a pristine copy of the stated commit. The full
resulting `verl/` tree was compared by file set, mode, and content against the
bundled snapshot: it contains nine content-modified files and one normalized
pathname. Upstream Verl remains Apache-2.0; its license and notice are under
`THIRD_PARTY_LICENSES/verl/`.
