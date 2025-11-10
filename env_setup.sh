#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/algorithms:$PROJECT_ROOT/src/functions:$PROJECT_ROOT/src/utils"
export PYTHONPATH