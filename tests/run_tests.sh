#!/usr/bin/env bash
# run_tests.sh — Execute the full test suite from the project root.
#
# Usage:
#   cd /path/to/ai_labor_research
#   bash tests/run_tests.sh [pytest-args]
#
# Examples:
#   bash tests/run_tests.sh                        # run all tests
#   bash tests/run_tests.sh -v                     # verbose output
#   bash tests/run_tests.sh -k test_data_quality   # run one module
#   bash tests/run_tests.sh --tb=short             # compact tracebacks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

echo "============================================================"
echo "  AI Labor Research — Test Suite"
echo "  Project root : $PROJECT_ROOT"
echo "  Python       : $(python3 --version)"
echo "============================================================"
echo ""

python3 -m pytest tests/ \
    --tb=short \
    -v \
    "$@"
