#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Usage" "$0" "<result_dir>"
    exit
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
readonly SCRIPT_DIR
readonly result_dir="$1"

for fold_dir in "$result_dir"/?; do
    (
        echo "$fold_dir"
        python3 "$SCRIPT_DIR"/pca_spine.py "$fold_dir"/train_val.ndjson "$fold_dir" --mirror
        for image_type in ap_whole lateral_whole; do
            outdir="$fold_dir"/auto/$image_type
            mkdir -p "$outdir"
            python3 "$SCRIPT_DIR"/apply_model.py hm2lm "$fold_dir"/y_pred_$image_type.mha "$fold_dir"/test_$image_type.ndjson "$outdir" --save_ndjson "$outdir"/all_heatmap_keypoints.ndjson
            python3 "$SCRIPT_DIR"/fit_spine_model.py "$outdir"/all_heatmap_keypoints.ndjson "$fold_dir"/y_pred_$image_type.mha --output "$outdir"/all_pca_keypoints.ndjson -m "$fold_dir"/pca_${image_type}_L4.npz "$fold_dir"/pca_${image_type}_L5.npz --sigma 0.5 --max_iter 400 --l2_reg 0.001 --device cpu --skipped "$outdir"/skipped.txt
        done
    ) &
done
wait
echo 'end applying and fitting:' "$(date)"
mkdir -p "$result_dir"/{auto,reference}{,/keypoints}/{ap_whole,lateral_whole}
echo 'collect ndjson files from all folds and modify them'
for image_type in ap_whole lateral_whole; do
    cat "$result_dir"/?/auto/$image_type/all_pca_keypoints.ndjson >"$result_dir"/auto/keypoints/$image_type/all_pca_keypoints.ndjson
    python3 "$SCRIPT_DIR"/../script/modify_ai_annotation.py $image_type "$result_dir"/auto/keypoints/$image_type/all_pca_keypoints.ndjson --output "$result_dir"/auto/keypoints/$image_type/all_final_keypoints.ndjson
done
