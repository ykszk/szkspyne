#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Usage" "$0" "<result_dir>"
    exit
fi
result_dir="$1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

echo 'start:' "$(date)"
for fold_dir in "$result_dir"/?; do
    (
        echo "$fold_dir"
        python3 "$SCRIPT_DIR"/pca_spine.py "$fold_dir"/train_val.ndjson "$fold_dir"
        for image_type in ap_whole lateral_whole; do
            outdir="$fold_dir"/auto/$image_type
            mkdir -p "$outdir"
            python3 "$SCRIPT_DIR"/apply_model.py hm2lm "$fold_dir"/y_pred_$image_type.mha "$fold_dir"/test_$image_type.ndjson "$outdir" --save_heatmap
            python3 "$SCRIPT_DIR"/fit_spine_model.py "$outdir" --channel_first -m "$fold_dir"/pca_${image_type}_L4.npz -m "$fold_dir"/pca_${image_type}_L5.npz --sigma 10 1 --device cpu
        done
    ) &
done
wait
echo 'end applying and fitting:' "$(date)"
mkdir -p "$result_dir"/{auto,reference}{,/keypoints}/{ap_whole,lateral_whole}
echo 'collect json files from all folds'
for image_type in ap_whole lateral_whole; do
    cp "$result_dir"/?/auto/$image_type/*.json "$result_dir"/auto/keypoints/$image_type/
done
python3 "$SCRIPT_DIR"/../script/modify_ai_annotation.py "$result_dir"/auto/keypoints

HIDE="MajorApex APT CoronalBalance T1Tilt ClavicleAngle ShoulderHeight Pelvis FemoralHead PelvicObliquity LegLength SacralObliquity Mid/LowerThoracicKyphosis ProximalThoracicKyphosis SacralSlope PelvicIncidence PelvicRadiusAngle L5IncidenceAngle LumbosacralAngle"
for image_type in ap lateral; do
    # create reference
    grep ${image_type}_whole "$result_dir"/manual_annotation.ndjson | python3 "$SCRIPT_DIR"/../script/exp2lmjson.py - "$result_dir"/reference/keypoints/${image_type}_whole
    for subdir in auto reference; do
        echo measure and draw $image_type $subdir
        python3 "$SCRIPT_DIR"/measure_and_draw.py --hide $HIDE --height 1200 --points --image_type $image_type --parent_dir "$SCRIPT_DIR"/../../data/annotation/images/all --config "$SCRIPT_DIR"/../../data/annotation/config.yaml batch "$result_dir"/$subdir/keypoints/${image_type}_whole "$result_dir"/$subdir/${image_type}_whole/ --html --image .svg --measure
        python3 "$SCRIPT_DIR"/create_svg_catalog.py --title AI "$result_dir"/$subdir/${image_type}_whole/ "$result_dir"/$subdir/catalog_${image_type}_whole.html
    done
done
python3 "$SCRIPT_DIR"/../script/measured_jsons2csv.py "$result_dir"
INPUT_DIR="$result_dir" jupyter nbconvert "$SCRIPT_DIR"/../script/exp_summary.ipynb --no-input --execute --to html --output "$result_dir"/summary.html

python3 "$SCRIPT_DIR"/../script/pair_data.py "$result_dir"
INPUT_DIR="$result_dir" jupyter nbconvert "$SCRIPT_DIR"/../script/ap_summary.ipynb --no-input --execute --to html --output "$result_dir"/summary_bend.html

echo 'end measuring:' "$(date)"
