#!/usr/bin/env bash
if [ "$#" -lt 2 ]; then
    echo "Usage" "$0" "<result_dir> <ndjson> [--holdout]"
    exit
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
readonly SCRIPT_DIR
readonly result_dir="$1"
readonly ndjson_filename="$2"

readonly N_SPLIT=5

python3 "$SCRIPT_DIR"/preexp.py --fold "$N_SPLIT" --output "$result_dir" --ndjson "$ndjson_filename" #"$3"

for fold_dir in "$result_dir"/?; do
    echo "$fold_dir"
    python3 "$SCRIPT_DIR"/exp_single.py --clahe --encoder_name efficientnet-b4 --batch_size 6 --lr 0.001 --sigma 3 --max-epochs 128 --input "$fold_dir" || break
done
