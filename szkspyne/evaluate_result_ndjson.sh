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

readonly HIDE="CurveApex APT CoronalBalance T1Tilt ClavicleAngle ShoulderHeight Pelvis FemoralHead PelvicObliquity LegLength SacralObliquity Mid/LowerThoracicKyphosis ProximalThoracicKyphosis SacralSlope PelvicIncidence PelvicRadiusAngle L5IncidenceAngle LumbosacralAngle"
algo="no-cross"
# algo="max_diffs"
for image_type in ap lateral; do
    (
        # create reference
        grep ${image_type}_whole "$result_dir"/manual_annotation.ndjson | python3 "$SCRIPT_DIR"/../script/exp2lmjson.py - "$result_dir"/reference/keypoints/${image_type}_whole/all_final_keypoints.ndjson
        for subdir in auto reference; do
            echo measure and draw $image_type $subdir
            python3 "$SCRIPT_DIR"/measure_and_draw.py --hide $HIDE --height 1200 --points --numbers --image_type $image_type --parent_dir "$SCRIPT_DIR"/../../data/annotation/images/all --title "${subdir} ${image_type}" --config "$SCRIPT_DIR"/../../data/annotation/config.yaml --algo "$algo" --imagedb "$result_dir"/heatmap_images.sqlite3 batch "$result_dir"/$subdir/keypoints/${image_type}_whole/all_final_keypoints.ndjson "$result_dir"/$subdir/${image_type}_whole/ --html --ndjson "$result_dir"/$subdir/measures_${image_type}_whole.ndjson
            # python3 "$SCRIPT_DIR"/create_svg_catalog.py --title AI "$result_dir"/$subdir/${image_type}_whole/ "$result_dir"/$subdir/catalog_${image_type}_whole.html
        done
    ) &
done
wait

# closest endplates
python3 "$SCRIPT_DIR"/../script/find_endplate.py "$result_dir"
subdir="auto"
image_type="ap"
suffix="_w_closest_indices"
mkdir -p "$result_dir"/"$subdir"/"$image_type"_whole${suffix}
ndjson_w_indices="$result_dir"/auto/keypoints/ap_whole/all_final_keypoints${suffix}.ndjson
lmrs join --mode inner "$result_dir"/auto/keypoints/ap_whole/all_final_keypoints.ndjson "$result_dir"/auto/closest_indices.ndjson  > "$ndjson_w_indices"

algo="stored"
python3 "$SCRIPT_DIR"/measure_and_draw.py --hide "$HIDE" --height 1200 --points --numbers --image_type $image_type --parent_dir "$SCRIPT_DIR"/../../data/annotation/images/all --title "${subdir} ${image_type}" --config "$SCRIPT_DIR"/../../data/annotation/config.yaml --algo "$algo" --imagedb "$result_dir"/heatmap_images.sqlite3 batch "$ndjson_w_indices" "$result_dir"/$subdir/${image_type}_whole${suffix}/ --html --ndjson "$result_dir"/$subdir/measures_${image_type}_whole${suffix}.ndjson
readonly REUSE_CURVE=false
# reuse curve
if "${REUSE_CURVE}"; then
    subdir="auto"
    image_type="ap"
    suffix="_w_curve_indices"
    mkdir -p "$result_dir"/"$subdir"/"$image_type"_whole${suffix}
    algo="stored"
    ndjson_w_indices="$result_dir"/$subdir/keypoints/${image_type}_whole/all_final_keypoints${suffix}.ndjson
    cat "$result_dir"/reference/measures_ap_whole.ndjson | jq -c '{"curve_indices": .curve_indices, "filename": .filename}' | lmrs join --mode inner "$result_dir"/auto/keypoints/ap_whole/all_final_keypoints.ndjson - >"$ndjson_w_indices"
    python3 "$SCRIPT_DIR"/measure_and_draw.py --hide "$HIDE" --height 1200 --points --numbers --image_type $image_type --parent_dir "$SCRIPT_DIR"/../../data/annotation/images/all --title "${subdir} ${image_type}" --config "$SCRIPT_DIR"/../../data/annotation/config.yaml --algo "$algo" --imagedb "$result_dir"/heatmap_images.sqlite3 batch "$ndjson_w_indices" "$result_dir"/$subdir/${image_type}_whole${suffix}/ --html --ndjson "$result_dir"/$subdir/measures_${image_type}_whole${suffix}.ndjson
fi

suffix="_w_closest_indices"
python3 "$SCRIPT_DIR"/../script/measured_jsons2csv.py "$result_dir"
jupytext --to ipynb --output - "$SCRIPT_DIR"/../notebooks/exp_summary.py --update-metadata '{"title":"exp summary"}' | INPUT_DIR="$result_dir" NO_VIS=true jupyter nbconvert --stdin --to html --execute --no-input --output "$result_dir"/summary.html

jupytext --to ipynb --output - "$SCRIPT_DIR"/../notebooks/exp_summary.py --update-metadata '{"title":"exp summary w closest indices"}' | INPUT_DIR="$result_dir" NO_VIS=true DATA_SUFFIX=${suffix} jupyter nbconvert --stdin --to html --execute --no-input --output "$result_dir"/summary${suffix}.html

if "${REUSE_CURVE}"; then
    suffix="_w_curve_indices"
    jupytext --to ipynb --output - "$SCRIPT_DIR"/../notebooks/exp_summary.py --update-metadata '{"title":"exp summary w copied indices"}' | INPUT_DIR="$result_dir" NO_VIS=true DATA_SUFFIX=${suffix} jupyter nbconvert --stdin --to html --execute --no-input --output "$result_dir"/summary${suffix}.html
fi

# python3 "$SCRIPT_DIR"/../script/pair_data.py "$result_dir" || (echo "Failed at pair_data.py" && exit 1)
# INPUT_DIR="$result_dir" jupyter nbconvert "$SCRIPT_DIR"/../script/ap_summary.ipynb --no-input --execute --to html --output "$result_dir"/summary_flex.html

echo 'end measuring:' "$(date)"
