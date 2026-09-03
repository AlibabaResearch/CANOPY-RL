# Contributing

Thank you for helping improve CANOPY. This repository preserves a tested paper
reference baseline while allowing separately documented recipe ports, so
changes should preserve reproducibility and provenance.

## Before opening a change

- Do not rewrite the paper baseline implicitly. The audited source-layout reference is
  `19c6af5de10de2b5272c83c0e82aa715c8c621f3` plus
  `patches/verl-canopy.patch`.
- A compatibility contribution for another Verl revision must name the exact
  commit, explain required adaptations, and include import, resolved-config,
  and runtime smoke-test evidence. It must not be presented as paper-exact
  reproduction without corresponding end-to-end evidence.
- Do not commit datasets, model weights, checkpoints, unreviewed or protected
  prompts, trajectories, logs, container archives, credentials, personal
  paths, or internal hostnames. A public prompt may be included only after its
  source, immutable revision, license, distributed path, and modifications are
  recorded with the third-party inventory.
- If code is copied or adapted, record its source URL, immutable revision,
  copyright, license, distributed paths, and modifications in
  `THIRD_PARTY_COMPONENTS.yml` and retain the complete license text.
- Keep AppWorld protected content and derivatives outside the public tree.

## Checks

From the repository root, run:

```bash
python3 -m compileall -q recipe tools
python3 tools/check_public_release.py
find run_scripts recipe/appworld/env_server -name '*.sh' -print0 | xargs -0 -n 1 bash -n
```

If a change touches `patches/verl-canopy.patch`, also verify it against a clean
copy of the fixed commit:

```bash
git -C /path/to/clean-verl apply --check /path/to/CANOPY/patches/verl-canopy.patch
```

For runtime changes, include the resolved configuration, environment version,
hardware layout, dataset/model revision, and the smallest relevant test result
in the pull request. Never attach protected or confidential inputs to a public
issue or pull request.
