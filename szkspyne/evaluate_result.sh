#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Usage" "$0" "<result_dir>"
    exit
fi
result_dir="$1"
for i in $(seq 0 4); do
    echo fold "$i"
    fold_dir=$result_dir/$i
    python3 pca_spine.py "$fold_dir"/train_val.ndjson "$fold_dir"
    for image_type in ap_whole lateral_whole; do
        outdir="$fold_dir"/auto/$image_type
        mkdir -p "$outdir"
        python3 apply_model.py hm2lm "$fold_dir"/y_pred_$image_type.mha "$fold_dir"/test_$image_type.ndjson "$outdir" --save_heatmap
        python3 fit_spine_model.py "$outdir" --channel_first -m "$fold_dir"/pca_${image_type}_L4.npz -m "$fold_dir"/pca_${image_type}_L5.npz --device cpu
    done
done
