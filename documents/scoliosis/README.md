# Automatic Measurement of Spinal Alignment
This document describes how to train a model for extracting landmark points from an x-ray image and measure spinal alignment parameters (e.g. Cobb angles).

## Requirements
- uv: python project manager
    - follow the [instruction](https://docs.astral.sh/uv/getting-started/installation/) for installation
- rust: for building required command line programs
    - follow the [instruction](https://rust-lang.org/tools/install/) for installation

### Commands for pre/post-processing
- [jqlang/jq: Command-line JSON processor](https://github.com/jqlang/jq)
    - follow the official document for installation
- [ykszk/scolrs](https://github.com/ykszk/scolrs)
- [ykszk/labelme-rs](https://github.com/ykszk/labelme-rs)
```sh
cargo install --git https://github.com/ykszk/scolrs --tag v0.1.1 --features gdcm
cargo install --git https://github.com/ykszk/labelme-rs --tag v0.14.1
```

## Setup
```sh
# download python dependencies
uv sync
# activate the virtual environment
source .venv/bin/activate
```

## Dataset
Create an annotated dataset using [labelme](https://github.com/wkentaro/labelme).
Add `ap_whole` flags for frontal/coronal images and `lateral_whole` flags for lateral/sagittal images.

See `sample_data` directory for examples.

## Train

### (Create dummy dataset)
Create dummy dataset by copying data in `sample_data`.
```sh
bash ./documents/scoliosis/create_dummy.sh
```

### Compile dataset
```sh
lmrs ndjson documents/scoliosis/dummy | lmrs swap - $PWD/documents/scoliosis/dummy | python script/compile_labelme_annotations.py | python3 script/insert_group_id.py insert_group_id anchor > data/annotations_.ndjson
# add `group_id` to each line. `group_id`s are used as patient IDs and dataset splits are based on PatientIDs.
jq -c -n '
  foreach inputs as $line (
    0;
    . + 1;
    . as $n | $line | .content.group_id = $n
  )
' data/annotations_.ndjson > data/annotations.ndjson
```

### Run training script
Training logs to [W&B](https://wandb.ai/); run `wandb login` first, or set `WANDB_MODE=offline` to skip.
```sh
export EXPOUT=/tmp/szkspyne/result && mkdir -p $EXPOUT && bash -x ./szkspyne/run_kfold.sh $EXPOUT data/annotations.ndjson --holdout 
```
The script splits the dataset into train and test subsets, trains the model using the training dataset, and applies the trained model to the test dataset.

## Evaluate
```sh
RESULT_DIR=/tmp/szkspyne/result doit -n 4 -f szkspyne/dodo_postprocess.py summary
```
The script measures spinal parameters and produces `summary_ap_whole.html` and `summary_lateral_whole.html` in the `RESULT_DIR/postprocess` that summarizes measurements. The evaluation process ends with error but that's due to the lack of pre/post Op flags in the sample data.