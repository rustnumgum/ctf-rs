#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd "$(dirname "$0")/.." && pwd)"
work_dir="$HOME/ctf-rs-work"
if [[ "$source_dir" == "$work_dir" ]]; then
  echo 'Run this sync script from the Windows-backed delivery repository.' >&2
  exit 1
fi
mkdir -p "$work_dir"
rsync -a --exclude=.git --exclude=target "$source_dir/" "$work_dir/"
printf 'Linux work copy: %s\n' "$work_dir"
