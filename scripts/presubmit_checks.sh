#!/bin/bash

set -uexo pipefail

readonly MY_PATH=$(cd $(dirname "$0") && pwd)

cd ${MY_PATH}/..

black --check src/

pyright src/

isort --check-only src/

python -m unittest src/**/*_test.py

echo All done 👍
