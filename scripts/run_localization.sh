#!/bin/bash

MODEL=${1:-"gpt-4o"}
START=${2:-0}
END=${3:-"-1"}

python eval.py localization \
    --model "$MODEL" \
    --start "$START" \
    --end "$END" \
    --error_yaml configs/error_types.yaml \
    --output_dir results \
    --verbose
