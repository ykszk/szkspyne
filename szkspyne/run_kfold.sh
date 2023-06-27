#!/usr/bin/env bash
if [ "$#" -lt 1 ]; then
    echo "Usage" "$0" "<result_dir> [--holdout]"
    exit
fi
result_dir="$1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

N_SPLIT=5

python3 "$SCRIPT_DIR"/preexp.py --fold 5 --output $result_dir $2 --ndjson "$SCRIPT_DIR"/../script/manual_annotation_wo_CLO.ndjson

for fold_dir in "$result_dir"/?; do
    echo "$fold_dir"
    python3 "$SCRIPT_DIR"/exp_single.py --clahe --encoder_name efficientnet-b4 --batch_size 6 --lr 0.001 --sigma 3 --max-epochs 128 --input "$fold_dir" || break
done
