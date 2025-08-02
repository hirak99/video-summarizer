#!/bin/bash

set -uexo pipefail

readonly MY_PATH=$(cd $(dirname "$0") && pwd)

cd ${MY_PATH}/..

black --check src/

pyright src/

isort --check-only src/

find -type f -iname '*_test.py' -exec python -m unittest {} +

echo All done 👍
