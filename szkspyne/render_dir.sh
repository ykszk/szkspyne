#!/usr/bin/env bash
if [ "$#" -ne 2 ]; then
    echo "Usage" "$0" "<svg_dir>" "<out_dir>"
    exit
fi
svg_dir=$1
out_dir=$2
for image_type in ap_whole lateral_whole; do
    for fn in "${svg_dir}"/"${image_type}"/*.json; do
        filename=$(basename -- "$fn")
        echo -n $filename " " && date
        stem="${filename%.*}"
        python3 ./measure_and_draw.py "$fn" ${out_dir}/$image_type/"${stem}".html --image ${out_dir}/$image_type/"${stem}".svg --config "data/config.yaml" --template template --points --title "$stem" --height 1200 --image_type ${image_type%_*}
    done
    break
done
