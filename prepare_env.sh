#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements.txt"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.11}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"


if command -v pyenv >/dev/null 2>&1; then
    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    eval "$(pyenv init -)"
    pyenv install -s "${PYTHON_VERSION}"
    PYTHON_BIN="$(pyenv prefix "${PYTHON_VERSION}")/bin/python"
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} not found and pyenv is not available."
    echo "Install python3.11/python3.11-venv or install pyenv."
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment in ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

ENV_PYTHON="${VENV_DIR}/bin/python"
ENV_PIP="${VENV_DIR}/bin/pip"

"${ENV_PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' || {
    echo "ERROR: ${VENV_DIR} must use Python 3.11. Recreate it with ${PYTHON_BIN} -m venv ${VENV_DIR}"
    exit 1
}

echo "Using Python: ${ENV_PYTHON}"
"${ENV_PIP}" install --upgrade pip setuptools wheel
"${ENV_PIP}" install -r "${REQ_FILE}"

echo
echo "Environment ready."
echo "Activate it with:"
echo "source ${VENV_DIR}/bin/activate"
