#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3.12}
VENV=${VENV:-$ROOT/.venv-readout-v2}

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  --requirement "$ROOT/requirements-readout-v2.txt"
"$VENV/bin/python" -m pip check

"$VENV/bin/python" - <<'PY'
import joblib
import numpy
import sklearn

print(f"joblib={joblib.__version__}")
print(f"numpy={numpy.__version__}")
print(f"scikit-learn={sklearn.__version__}")
PY
