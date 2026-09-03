#!/usr/bin/env bash
# Build the public ZIP from exactly the clean, SCA-scanned Git commit.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"

if [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]]; then
    echo "Refusing to package a dirty Git worktree." >&2
    exit 1
fi

python3 "${repo_root}/tools/check_public_release.py" --root "${repo_root}"

commit="$(git -C "${repo_root}" rev-parse HEAD)"
output_dir="${repo_root}/release"
output_zip="${output_dir}/canopy-public.zip"
checksum_file="${output_zip}.sha256"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/canopy-public.XXXXXX")"

cleanup() {
    case "${temp_dir}" in
        "${TMPDIR:-/tmp}"/canopy-public.*) rm -rf -- "${temp_dir}" ;;
        *) echo "Refusing to remove unexpected temporary path: ${temp_dir}" >&2 ;;
    esac
}
trap cleanup EXIT

temp_zip="${temp_dir}/canopy-public.zip"
git -C "${repo_root}" archive --format=zip --prefix=CANOPY/ --output="${temp_zip}" "${commit}"

mkdir -p -- "${output_dir}"
cp "${temp_zip}" "${output_zip}"

if command -v sha256sum >/dev/null 2>&1; then
    digest="$(sha256sum "${output_zip}" | awk '{print $1}')"
else
    digest="$(shasum -a 256 "${output_zip}" | awk '{print $1}')"
fi
printf '%s  %s\n' "${digest}" "$(basename "${output_zip}")" > "${checksum_file}"

echo "commit=${commit}"
echo "archive=${output_zip}"
echo "sha256=${digest}"
