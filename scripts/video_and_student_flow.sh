#!/bin/bash

set -uexo pipefail

readonly MY_PATH="$(dirname "$(realpath "$0")")"

# Process each video.
# Note: All arguments will be used here.
"$MY_PATH"/video_flow.sh "$@"

# Compile student highlights.
"$MY_PATH"/student_flow.sh
