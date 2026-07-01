cd documents/scoliosis
mkdir -p dummy
for i in {1..20}
do
    # copy image files
    cp sample_data/lateral.jpg dummy/lateral_$i.jpg
    cp sample_data/frontal.jpg dummy/frontal_$i.jpg
    # copy json files
    sed "s/lateral.jpg/lateral_$i.jpg/g" sample_data/lateral.json > dummy/lateral_$i.json
    sed "s/frontal.jpg/frontal_$i.jpg/g" sample_data/frontal.json > dummy/frontal_$i.json
    # cp sample_data/lateral.json dummy/lateral_$i.json
    # cp sample_data/frontal.json dummy/frontal_$i.json
done