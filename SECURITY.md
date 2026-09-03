# Security policy

## Supported versions

Security fixes to CANOPY's distributed source are applied to the current
`main` branch. The bundled tested Verl source is identified in
`patches/README.md`. An operator-selected external Verl/Python/GPU environment
is maintained and scanned by its operator and upstream providers; best-effort
recipe portability does not imply security support for that environment.

## Reporting a vulnerability

Please use GitHub private vulnerability reporting at
<https://github.com/AlibabaResearch/SignalCoverageRL/security/advisories/new>. Do not disclose a
suspected vulnerability, credential, private path, protected benchmark input,
or exploit in a public issue. If private reporting is not yet enabled, contact
the repository maintainers through the Alibaba open-source process before
sharing sensitive details.

## Runtime security notes

Both environments execute model-generated or repository-supplied code. Treat
all such code as untrusted.

## AppWorld service

- The HTTP service has no authentication or tenant isolation.
- The launcher binds to `127.0.0.1` by default. A non-local bind is refused
  unless `APPWORLD_ALLOW_REMOTE_BIND=1` is explicitly set.
- When binding a wildcard address on a trusted multi-node network, set
  `APPWORLD_ADVERTISE_IP` to the node's routable private address; never
  advertise `0.0.0.0` as a client endpoint.
- Use a private, isolated compute network with host firewall rules. Never expose
  the ports to the public Internet.
- Run the service under a dedicated, non-privileged account and avoid mounting
  secrets, SSH agents, cloud credentials, or writable source trees.
- `APPWORLD_NUM_SERVERS`, request sizes, session count, execution time, and
  worker memory should remain bounded.
- Evaluation output is accepted only below `APPWORLD_ALLOWED_OUTPUT_ROOT` to
  prevent arbitrary host-path writes.

## SWE containers

- Use rootless Podman where possible. Do not mount the host container socket,
  credentials, or unrelated writable directories.
- Host networking and `seccomp=unconfined` are not enabled by default in the
  cleaned client. Any local opt-in weakens isolation and requires a dedicated
  disposable worker.
- Public dependency mirrors are disabled by default and do not change the
  container network mode. If package downloads are required, enable only the
  relevant ecosystem gates and provide restricted egress on disposable,
  secret-free workers; preserve the default seccomp profile.
- Repository reset/cleanup commands are restricted to the in-container
  `/testbed` tree. Do not weaken that guard.
- Host-wide process/image garbage collection is not included. Do not add global
  `pkill`, container removal, or image-prune commands; they can terminate jobs
  or remove images belonging to other users on the same node.
- Image archives are resolved from `SWE_IMAGE_ROOT`; validate ownership and
  file permissions before extraction.

## Logging and network egress

Paper commands default to console-only logging. Do not add credentials to shell
scripts or tracked YAML. Put private runtime values in an untracked local
configuration and run `tools/check_public_release.py` before packaging.
