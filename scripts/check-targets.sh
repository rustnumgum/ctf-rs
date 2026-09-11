#!/usr/bin/env bash
# Cross-checks scripts/acceptance-targets.tsv against `cargo metadata` and
# against the crate's own unit tests:
#   - every `[[test]]` target (declared or auto-discovered from tests/*.rs)
#     must appear in the manifest as class mpi, local, or excluded, and every
#     manifest row of those three classes must name a target that actually
#     exists.
#   - every class `lib` row holds a `cargo test --lib <filter>` filter
#     string rather than a target name, and must match at least one unit
#     test (`cargo test --lib <filter> -- --list`), so a filter left behind
#     by a deleted or renamed test fails the check instead of silently
#     running zero tests.
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

lib_filters="$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    for row in reader:
        if row["class"] == "lib":
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

# Every lib filter must match at least one unit test. A filter matching
# zero tests exits 0 and prints "0 tests, 0 benchmarks" from cargo/libtest,
# so this cannot be read off $? alone; count the actual `name: test` lines.
empty_lib_filters=()
if [[ -n "$lib_filters" ]]; then
  while IFS= read -r filter; do
    [[ -z "$filter" ]] && continue
    list_output="$(cargo test --lib "$filter" -- --list 2>&1 || true)"
    match_count="$(grep -c ': test$' <<<"$list_output" || true)"
    if [[ "${match_count:-0}" -eq 0 ]]; then
      empty_lib_filters+=("$filter")
    fi
  done <<<"$lib_filters"
fi

if (( ${#empty_lib_filters[@]} > 0 )); then
  echo "check-targets: lib filters (cargo test --lib <filter>) matching no unit test:" >&2
  for f in "${empty_lib_filters[@]}"; do
    echo "  $f" >&2
  done
  status=1
fi

total="$(wc -l <<<"$metadata_names" | tr -d ' ')"
lib_count="$(wc -w <<<"$lib_filters" | tr -d ' ')"
if [[ $status -eq 0 ]]; then
  echo "check-targets: OK, $total cargo test targets and $lib_count lib filters all accounted for in $manifest"
fi
exit "$status"
