#!/bin/sh
# Format Python code using isort, docformatter and black

set -e
isort --profile black mmocc
docformatter --black --in-place -r mmocc
black mmocc