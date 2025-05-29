#%%
import pydicom
import matplotlib.pyplot as plt
from szkspyne.coord import Spine
from szkspyne import labelme_utils
import numpy as np
import pandas as pd

ndjson_filename = '../script/group_id_dicom_manual_annotation.ndjson'
pairing_filename = '../script/pre_post_pairs.csv'

# with open(ndjson_filename, 'r') as f:
#     lines = f.readlines()
#     shape_lines = [labelme_utils.ShapeDictLine.model_validate_json(line) for line in lines]
shape_lines = labelme_utils.ShapeDictLine.from_ndjson(ndjson_filename)

print(len(shape_lines))
#%%
df_pairs = pd.read_csv(pairing_filename)
df_pairs.iloc[0], shape_lines[0].filename
# dictionary with keys of pre_op and values of post_op
dict_pairs = {row['pre_op']: row['post_op'] for _, row in df_pairs.iterrows()}
print(len(dict_pairs))

#%%
dict_shape_lines = {shape_line.filename: shape_line for shape_line in shape_lines}

#%%
shape_pairs = [(dict_shape_lines[pre_op], dict_shape_lines[post_op]) for pre_op, post_op in dict_pairs.items()]
shape_pairs[0]
#%%
from pathlib import Path
from PIL import Image
from skimage.transform import rescale
from tqdm import tqdm
import joblib
from utils import load_image, save_as_img

do_plot = False

outdir = Path('../../data/paired_dataset')
outdir.mkdir(exist_ok=True, parents=True)
image_outdir = outdir / 'images'
image_outdir.mkdir(exist_ok=True, parents=True)

def make_pair(shape_pair):
    shape_dict1, shape_dict2 = shape_pair
    shape_dict1 = shape_dict1.content
    shape_dict2 = shape_dict2.content

    spine1 = Spine.from_shape_dict(shape_dict1)
    spine2 = Spine.from_shape_dict(shape_dict2)

    percpercentile=4
    img1 = load_image(shape_dict1.imagePath, percentile=percpercentile)
    img2 = load_image(shape_dict2.imagePath, percentile=percpercentile)

    if do_plot:
        _, [ax1, ax2] = plt.subplots(1, 2)
        ax1.imshow(img1, cmap='gray')
        ax1.plot(spine1.vertebrae[:, :, 0].ravel(), spine1.vertebrae[:, :, 1].ravel(), 'o', markersize=3, color='blue')
        ax1.set_title('Pre OP')
        ax2.imshow(img2, cmap='gray')
        ax2.plot(spine2.vertebrae[:, :, 0].ravel(), spine2.vertebrae[:, :, 1].ravel(), 'o', markersize=3, color='red')
        ax2.set_title('Post OP')
        plt.show()

    aligned_spines = Spine.align_spines(spine1, img1, spine2, img2)
    # aligned_spines = Spine.align_spines(spine2, img2, spine1, img1)
    aligned_spine1, aligned_img1, aligned_spine2, aligned_img2  = aligned_spines
    if do_plot:
        _, [ax1, ax2] = plt.subplots(1, 2)
        ax1.imshow(aligned_img1, cmap='gray')
        ax1.plot(aligned_spine1[:, :, 0].ravel(), aligned_spine1[:, :, 1].ravel(), 'o', markersize=3, color='blue')
        ax1.set_title('Pre OP')
        ax2.imshow(aligned_img2, cmap='gray')
        ax2.plot(aligned_spine2[:, :, 0].ravel(), aligned_spine2[:, :, 1].ravel(), 'o', markersize=3, color='red')
        ax2.set_title('Post OP')

    # downsample
    scale = 0.5
    aligned_spine1 = rescale(aligned_spine1, scale, anti_aliasing=True)
    aligned_spine2 = rescale(aligned_spine2, scale, anti_aliasing=True)
    aligned_img1 = rescale(aligned_img1, scale, anti_aliasing=True)
    aligned_img2 = rescale(aligned_img2, scale, anti_aliasing=True)

    if do_plot:
        _, [ax1, ax2] = plt.subplots(1, 2)
        ax1.imshow(aligned_img1, cmap='gray')
        ax1.set_title('Pre OP')
        ax2.imshow(aligned_img2, cmap='gray')
        ax2.set_title('Post OP')
        plt.show()

    pre_filename = f'{Path(shape_dict1.imagePath).stem}.jpg'
    save_as_img(image_outdir / pre_filename, aligned_img1)
    post_filename = f'{Path(shape_dict2.imagePath).stem}.jpg'
    save_as_img(image_outdir / post_filename, aligned_img2)

    d_pre = {'filename': pre_filename, 'vertebrae': aligned_spine1.tolist() }
    d_post = {'filename': post_filename, 'vertebrae': aligned_spine2.tolist() }
    d_pair = {
        'pre_op': d_pre,
        'post_op': d_post
    }
    return d_pair

# output_pairs = []
# for shape_pair in tqdm(shape_pairs):
# use joblib
output_pairs = joblib.Parallel(n_jobs=8)(joblib.delayed(make_pair)(shape_pair) for shape_pair in tqdm(shape_pairs))

import json
with open(outdir / 'paired_dataset.ndjson', 'w') as f:
    for pair in output_pairs:
        f.write(json.dumps(pair) + '\n')
