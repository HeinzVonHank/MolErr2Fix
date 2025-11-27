#!/bin/bash

MODEL=${1:-"gpt-4o"}
EVAL_MODEL=${2:-"gpt-4o"}
START=${3:-0}
END=${4:-"-1"}

python eval.py revision \
    --model "$MODEL" \
    --eval_model "$EVAL_MODEL" \
    --start "$START" \
    --end "$END" \
    --output_dir results \
    --verbose
