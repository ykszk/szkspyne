# Cervical radiographic measurement and segmental motion assessment
This document describes how to train a model for extracting landmark points from an cervical x-ray image and measure cervical alignment (e.g. intervertebral angles) and range of motion parameters.

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
cargo install --git https://github.com/ykszk/scolrs --tag v0.1.3 --features gdcm
cargo install --git https://github.com/ykszk/labelme-rs --tag v0.14.4
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
See `sample_data` directory for examples.

## Train

### (Create dummy dataset)
Create dummy dataset by copying data in `sample_data`.
```sh
bash ./documents/neck/create_dummy.sh
```

### Compile dataset
```sh
lmrs ndjson documents/neck/dummy | lmrs swap - --prefix documents/neck/dummy | python script/compile_labelme_annotations.py | jq -c ' . + {"image_type": "lateral_whole"}' | python3 script/insert_group_id.py neck_group_id neck_anchor > data/annotations.ndjson
python documents/neck/sort_poses.py data/annotations.ndjson script/neck_study_poses.ndjson
```

### Run training script
Training logs to [W&B](https://wandb.ai/); run `wandb login` first, or set `WANDB_MODE=offline` to skip.
```sh
export EXPOUT=/tmp/szkspyne/result && mkdir -p $EXPOUT && bash -x ./szkspyne/run_kfold.sh $EXPOUT data/annotations.ndjson --holdout 
source script/neck_vars.sh; export EXPOUT=/tmp/szkspyne/result && mkdir -p $EXPOUT && bash -ex ./szkspyne/neck_run_kfold.sh $EXPOUT data/annotations.ndjson --holdout 
```
The script splits the dataset into train and test subsets, trains the model using the training dataset, and applies the trained model to the test dataset.

## Evaluate
```sh
source script/neck_vars.sh && export RESULT_DIR=/tmp/szkspyne/result doit -n 4 -f szkspyne/dodo_neck.py train_measure ref_draw draw_ai summary
```
The script measures cervical parameters and produces `summary.html` in the `RESULT_DIR` that summarizes measurements.