#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec sky launch -c tactile-hardness sky/compute.yaml "$@"
