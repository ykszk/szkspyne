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
        infile="$fold_dir"/train_val.ndjson
        outfile="$fold_dir"/pca_ap_whole_L4.npz
        if [ $infile -nt $outfile ]; then
            python3 "$SCRIPT_DIR"/pca_spine.py $infile "$fold_dir" --mirror
        fi
        for image_type in ap_whole lateral_whole; do
            outdir="$fold_dir"/auto/$image_type
            mkdir -p $outdir

            infile="$fold_dir"/y_pred_$image_type.mha
            outfile="$outdir"/all_heatmap_keypoints.ndjson
            if [ $infile -nt $outfile ]; then
                python3 "$SCRIPT_DIR"/heatmap2labelme.py $infile "$fold_dir"/test_$image_type.ndjson "$outdir" --save_ndjson $outfile
            fi

            infile="$fold_dir"/y_pred_$image_type.mha
            outfile="$result_dir"/heatmap_images.sqlite3
            if [ $infile -nt $outfile ]; then
                python3 "$SCRIPT_DIR"/../script/hm2db.py $infile "$outdir"/all_heatmap_keypoints.ndjson $outfile
            fi

            infile="$outdir"/all_heatmap_keypoints.ndjson
            outfile="$outdir"/all_pca_keypoints.ndjson
            if [ $infile -nt $outfile ]; then
                python3 "$SCRIPT_DIR"/fit_spine_model.py $infile "$fold_dir"/y_pred_$image_type.mha --output $outfile -m "$fold_dir"/pca_${image_type}_L4.npz "$fold_dir"/pca_${image_type}_L5.npz "$fold_dir"/pca_${image_type}_L6.npz --sigma 1 --max_iter 400 --l2_reg 0.001 --device cpu --skipped "$outdir"/skipped.txt --jobs 8
            fi
        done
    ) &
done
wait
echo 'end applying and fitting:' "$(date)"
mkdir -p "$result_dir"/{auto,reference}{,/keypoints}/{ap_whole,lateral_whole}

# Extract flags
cat "$result_dir"/manual_annotation.ndjson | jq -c '.filepath |= (split("/")[-1] |sub(".jpg";""))' | jq -c '{"flags": .flags, "filename": .filepath}' | jq -c '.flags = (.flags - ["L4","L5","L6"])' | jq -c '.flags |= reduce .[] as $flag ({}; .[$flag] = true)' > "$result_dir"/flags.ndjson

echo 'collect ndjson files from all folds and modify them'
for image_type in ap_whole lateral_whole; do
    infile="$result_dir"/0/auto/$image_type/all_pca_keypoints.ndjson
    outfile="$result_dir"/auto/keypoints/$image_type/all_final_keypoints.ndjson
    if [ $infile -nt $outfile ]; then
        cat "$result_dir"/?/auto/$image_type/all_pca_keypoints.ndjson >"$result_dir"/auto/keypoints/$image_type/all_pca_keypoints.ndjson
        python3 "$SCRIPT_DIR"/../script/modify_ai_annotation.py $image_type "$result_dir"/auto/keypoints/$image_type/all_pca_keypoints.ndjson --output - | jq -c 'del(.pca)' | lmrs join --mode left - "$result_dir"/flags.ndjson > $outfile
    fi
done
