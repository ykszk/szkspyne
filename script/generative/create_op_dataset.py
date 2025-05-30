#%%
from szkspyne import labelme_utils
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Create Operation datset')
parser.add_argument('--ndjson_filename', type=str, default='../../script/group_id_dicom_manual_annotation.ndjson',
                    help='Path to the NDJSON file containing DICOM annotations')
parser.add_argument('--pairing_filename', type=str, default='../../script/pre_post_pairs.csv',
                    help='Path to the CSV file containing pre and post operation pairs')
parser.add_argument('--op_params_filename', type=str, default='../../script/op_params.ndjson',
                    help='Path to the ndjson file containing operation parameters')
parser.add_argument('--src_img_dir', type=str, default='../../../data/generative_dataset/cyclegan_prepost/all_images',
                    help='Source directory containing images for the dataset')

parser.add_argument('--outdir', type=str, default='../../../data/generative_dataset/op_dataset/data',
                    help='Output directory the dataset')
# args = parser.parse_args()
args, _ = parser.parse_known_args()

ndjson_filename = args.ndjson_filename

shape_lines = labelme_utils.ShapeDictLine.from_ndjson(ndjson_filename)
df_pairs = pd.read_csv(args.pairing_filename)
df_pairs.head()
# %%
df_pairs = pd.read_csv(args.pairing_filename)

# %%
from szkspyne.type.op import OpParamsLine
        
op_params_lines = OpParamsLine.from_ndjson(args.op_params_filename)
op_params_lines[0]

# %%
op_params_dict = {Path(line.filename).name: line.content for line in op_params_lines}

#%%
shape_lines = labelme_utils.ShapeDictLine.from_ndjson(ndjson_filename)
dict_shape_lines = {Path(shape_line.filename).name: shape_line for shape_line in shape_lines}
dict_pairs = {row['pre_op']: row['post_op'] for _, row in df_pairs.iterrows()}
shape_pairs = [(dict_shape_lines[Path(pre_op).name], dict_shape_lines[Path(post_op).name]) for pre_op, post_op in dict_pairs.items()]
shape_pairs[0]

#%%
from szkspyne.coord import Spine
import json

outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True, parents=True)

import shutil
from tqdm import tqdm

src_img_dir = Path(args.src_img_dir)

for idx, row in tqdm(df_pairs.iterrows()):
    pre_op = Path(row['pre_op']).name
    post_op = Path(row['post_op']).name
    if pre_op not in op_params_dict or post_op not in op_params_dict:
        print(f'Missing parameters for {pre_op} or {post_op}')
        continue
    if pre_op == post_op:
        print(f'Skipping same operation pair: {pre_op}')
        continue
    pre_op_params = op_params_dict[pre_op]
    post_op_params = op_params_dict[post_op]

    pre_op_shape = dict_shape_lines[pre_op].content
    post_op_shape = dict_shape_lines[post_op].content

    pre_stem = Path(pre_op_shape.imagePath).stem
    post_stem = Path(post_op_shape.imagePath).stem
    # Copy images from source directory
    pre_img_src = src_img_dir / f'{pre_stem}.jpg'
    post_img_src = src_img_dir / f'{post_stem}.jpg'

    try:
        shutil.copy(pre_img_src, outdir / f'{pre_stem}.jpg')
        shutil.copy(post_img_src, outdir / f'{post_stem}.jpg')
    except FileNotFoundError as e:
        print(f'Error copying images: {e}')
        continue

    pre_op_dict = pre_op_params.model_dump()
    post_op_dict = post_op_params.model_dump()
    pre_op_dict['imagePath'] = f'{pre_stem}.jpg'
    pre_op_dict['PrePostImagePath'] = f'{post_stem}.jpg'
    post_op_dict['imagePath'] = f'{post_stem}.jpg'
    post_op_dict['PrePostImagePath'] = f'{pre_stem}.jpg'

    # save post params in pre-op file and vice versa
    with open(outdir / f'{post_stem}.json', 'w') as f:
        json.dump(pre_op_dict, f, indent=4)
    with open(outdir / f'{pre_stem}.json', 'w') as f:
        json.dump(post_op_dict, f, indent=4)

    # break
# %%
