#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/time -v -o "$1/rank-$OMPI_COMM_WORLD_RANK.time" "$2"
