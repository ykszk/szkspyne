#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Usage" "$0" "<result_dir>"
    exit
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
readonly SCRIPT_DIR
readonly result_dir="$1"

export LOGLEVEL=INFO
echo 'start:' "$(date)"
"$SCRIPT_DIR"/apply_pca_fitting.sh "$result_dir"

readonly jq_remove_flex='. | select(.flags."除外" or .flags."左屈" or .flags."右屈" or .flags."先天性側彎"|not)'
readonly HIDE="CurveApex APT CoronalBalance T1Tilt ClavicleAngle ShoulderHeight Pelvis FemoralHead PelvicObliquity LegLength SacralObliquity Mid/LowerThoracicKyphosis ProximalThoracicKyphosis SacralSlope PelvicIncidence PelvicRadiusAngle L5IncidenceAngle LumbosacralAngle"
export HIDE
algo="no-cross2"
# algo="max_diffs"
for image_type in ap lateral; do
    (
        # create reference
        grep ${image_type}_whole "$result_dir"/manual_annotation.ndjson | python3 "$SCRIPT_DIR"/../script/exp2lmjson.py - "$result_dir"/reference/keypoints/${image_type}_whole/all_final_keypoints.ndjson
        for subdir in auto reference; do
            echo measure and draw $image_type $subdir
            ndjson="$result_dir"/$subdir/keypoints/${image_type}_whole/all_final_keypoints.ndjson
            ndjson_curves="$result_dir"/$subdir/curve_indices_${image_type}_whole.ndjson
            cat $ndjson | jq -c "$jq_remove_flex" | python3 "$SCRIPT_DIR"/determine_curves.py - $ndjson_curves --algo $algo
            cat $ndjson | jq -c "$jq_remove_flex" | lmrs join --mode left - $ndjson_curves | python3 "$SCRIPT_DIR"/measure_and_draw.py --hide $HIDE --height 1200 --points --numbers --image_type $image_type --parent_dir "$SCRIPT_DIR"/../../data/annotation/images/all --title "${subdir} ${image_type}" --config "$SCRIPT_DIR"/../../data/annotation/config.yaml --imagedb "$result_dir"/heatmap_images.sqlite3 batch - "$result_dir"/$subdir/${image_type}_whole/ --html --ndjson "$result_dir"/$subdir/measures_${image_type}_whole.ndjson
        done
    ) &
done
wait

# find best matching curve sets
python3 "$SCRIPT_DIR"/../script/find_best_curve_set.py "$result_dir"

# closest endplates
python3 "$SCRIPT_DIR"/../script/find_endplate.py "$result_dir"
subdir="auto"
image_type="ap"
for variant in best_indices closest_indices; do
    suffix=_w_"$variant"
    mkdir -p "$result_dir"/"$subdir"/"$image_type"_whole${suffix}
    ndjson_w_indices="$result_dir"/auto/keypoints/ap_whole/all_final_keypoints${suffix}.ndjson
    lmrs join --mode inner "$result_dir"/auto/keypoints/ap_whole/all_final_keypoints.ndjson "$result_dir"/auto/"$variant".ndjson "$result_dir"/auto/curve_indices_${image_type}_whole.ndjson >"$ndjson_w_indices"

    cat "$ndjson_w_indices" | jq -c "$jq_remove_flex" | python3 "$SCRIPT_DIR"/measure_and_draw.py --hide $HIDE --height 1200 --points --numbers --image_type $image_type --parent_dir "$SCRIPT_DIR"/../../data/annotation/images/all --title "${subdir} ${image_type}" --config "$SCRIPT_DIR"/../../data/annotation/config.yaml --imagedb "$result_dir"/heatmap_images.sqlite3 batch - "$result_dir"/$subdir/${image_type}_whole${suffix}/ --html --ndjson "$result_dir"/$subdir/measures_${image_type}_whole${suffix}.ndjson
done

python3 "$SCRIPT_DIR"/../script/measured_jsons2csv.py "$result_dir"

jupytext --to ipynb --output - "$SCRIPT_DIR"/../notebooks/exp_summary.py --update-metadata '{"title":"exp summary"}' | INPUT_DIR="$result_dir" NO_VIS=true jupyter nbconvert --stdin --to html --execute --no-input --output "$result_dir"/summary.html

for variant in best_indices closest_indices; do
    suffix=_w_"$variant"
    jupytext --to ipynb --output - "$SCRIPT_DIR"/../notebooks/exp_summary.py --update-metadata "{\"title\":\"exp summary w $variant indices\"}" | INPUT_DIR="$result_dir" NO_VIS=true DATA_SUFFIX=${suffix} jupyter nbconvert --stdin --to html --execute --no-input --output "$result_dir"/summary${suffix}.html
done
# python3 "$SCRIPT_DIR"/../script/pair_data.py "$result_dir" || (echo "Failed at pair_data.py" && exit 1)
# INPUT_DIR="$result_dir" jupyter nbconvert "$SCRIPT_DIR"/../script/ap_summary.ipynb --no-input --execute --to html --output "$result_dir"/summary_flex.html

echo 'end measuring:' "$(date)"
