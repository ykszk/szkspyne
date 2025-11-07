
# cd /poseidon/users/suzuki/projects/scoliosis/data
for image_type in ap_whole lateral_whole; do
    echo ${image_type}
    cat annotation/annotated/*/valid_${image_type}.ndjson | jq -c '. | select(.content.flags."除外" or .content.flags."側面右向き"|not)' | jq -c '.content.flags."'${image_type}'" |= true' | lmrs shapeshift - c2p | python ../code/script/compile_labelme_annotations.py --check_image >/tmp/${image_type}_manual_annotation.ndjson
done
cat /tmp/ap_whole_manual_annotation.ndjson /tmp/lateral_whole_manual_annotation.ndjson >../code/script/manual_annotation.ndjson
cat ../code/script/manual_annotation.ndjson | python3 ../code/script/insert_group_id.py insert_group_id anchor >../code/script/group_id_manual_annotation.ndjson
cd ../code/script
cat manual_annotation.ndjson | jq -c '.content.imagePath |= sub("\\.[^.]*$"; ".dcm")' | jq -c '.content.imagePath = "/poseidon/users/suzuki/projects/scoliosis/data/dicom/selected/resized_all/" + (.content.imagePath | split("/")[-1])' >dicom_manual_annotation.ndjson
cat dicom_manual_annotation.ndjson | python3 insert_group_id.py insert_group_id anchor > group_id_dicom_manual_annotation.ndjson
