cd documents/neck
mkdir -p dummy
for i in {1..10}
do
    for image in sample_data/*.jpg
    do
        stem=$(basename $image .jpg)
        # copy image
        cp $image dummy/dummy${i}-${stem}.jpg
        # copy corresponding json
        sed "s/${stem}.jpg/dummy${i}-${stem}.jpg/g" sample_data/${stem}.json > dummy/dummy${i}-${stem}.json
    done
done
