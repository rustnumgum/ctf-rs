#!/usr/bin/env bash
# Cross-checks scripts/acceptance-targets.tsv against `cargo metadata`: every
# `[[test]]` target (declared or auto-discovered from tests/*.rs) must appear
# in the manifest as class mpi, local, or excluded, and every manifest row of
# those three classes must name a target that actually exists. class lib rows
# hold `cargo test --lib` filter strings, not target names, and are not
# checked here.
set -euo pipefail
cd "$(dirname "$0")/.."

manifest="scripts/acceptance-targets.tsv"
if [[ ! -f "$manifest" ]]; then
  echo "check-targets: missing $manifest" >&2
  exit 1
fi

metadata_names="$(cargo metadata --no-deps --format-version 1 |
  python3 -c '
import json, sys
meta = json.load(sys.stdin)
names = set()
for pkg in meta["packages"]:
    for t in pkg["targets"]:
        if "test" in t["kind"]:
            names.add(t["name"])
print("\n".join(sorted(names)))
')"

manifest_names="$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    for row in reader:
        if row["class"] in ("mpi", "local", "excluded"):
            print(row["target"])
' "$manifest")"

missing="$(comm -23 <(sort -u <<<"$metadata_names") <(sort -u <<<"$manifest_names"))"
extra="$(comm -13 <(sort -u <<<"$metadata_names") <(sort -u <<<"$manifest_names"))"

status=0
if [[ -n "$missing" ]]; then
  echo "check-targets: test targets missing from $manifest (not listed as mpi/local/excluded):" >&2
  echo "$missing" | sed 's/^/  /' >&2
  status=1
fi
if [[ -n "$extra" ]]; then
  echo "check-targets: manifest rows (class mpi/local/excluded) name no existing target:" >&2
  echo "$extra" | sed 's/^/  /' >&2
  status=1
fi

total="$(wc -l <<<"$metadata_names" | tr -d ' ')"
if [[ $status -eq 0 ]]; then
  echo "check-targets: OK, $total cargo test targets all accounted for in $manifest"
fi
exit "$status"
