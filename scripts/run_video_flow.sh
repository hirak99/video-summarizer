#!/bin/bash

set -uexo pipefail

readonly MY_PATH="$(dirname "$(realpath "$0")")"

cd "$MY_PATH"

python -m src.video_understanding.video_flow "$@"
