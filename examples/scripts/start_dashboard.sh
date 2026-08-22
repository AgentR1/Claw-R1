#!/usr/bin/env bash
set -euo pipefail
conda run -n "${CONDA_ENV:-steppo}" python -m dashboard.backend.server --config "${DASHBOARD_CONFIG:-dashboard/config.example.yaml}" "$@"
