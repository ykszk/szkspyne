#!/usr/bin/env bash
if [ "$#" -lt 2 ]; then
    echo "Usage" "$0" "<result_dir> <ndjson> [--holdout]"
    exit
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
readonly SCRIPT_DIR
readonly result_dir="$1"
readonly ndjson_filename="$2"

readonly N_SPLIT=10

# check if --holdout is set
if [ "$#" -eq 3 ]; then
    if [ "$3" != "--holdout" ]; then
        echo "Usage" "$0" "<result_dir> <ndjson> [--holdout]"
        exit
    fi
    echo "Holdout mode"
    python3 "$SCRIPT_DIR"/preexp.py --fold "$N_SPLIT" --output "$result_dir" "$ndjson_filename" --holdout
else
    echo "Kfold mode"
    python3 "$SCRIPT_DIR"/preexp.py --fold "$N_SPLIT" --output "$result_dir" "$ndjson_filename"
fi

for fold_dir in "$result_dir"/folds/*; do
    echo "$fold_dir"
    python3 "$SCRIPT_DIR"/exp_single.py --encoder_name efficientnet-b4 --batch_size 6 --lr 0.001 --sigma 3 --max-epochs 128 --input "$fold_dir"
done
