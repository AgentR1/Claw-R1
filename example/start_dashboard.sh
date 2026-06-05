#!/usr/bin/env bash
set -euo pipefail

conda run -n steppo python -m dashboard.backend.server \
  --config dashboard/config.example.yaml \
  "$@"
