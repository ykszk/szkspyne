#%%
from typing import List
from szkspyne import labelme_utils
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Create CycleGAN dataset from DICOM annotations')
parser.add_argument('--ndjson_filename', type=str, default='../../script/group_id_dicom_manual_annotation.ndjson',
                    help='Path to the NDJSON file containing DICOM annotations')
parser.add_argument('--image_outdir', type=str, default='../../../data/generative_dataset/cyclegan_prepost',
                    help='Output directory for the images')
# args = parser.parse_args()
args, _ = parser.parse_known_args()

ndjson_filename = args.ndjson_filename
image_outdir = Path(args.image_outdir)

shape_lines = labelme_utils.ShapeDictLine.from_ndjson(ndjson_filename)
# %%
import pandas as pd

ban_flags = set(['右屈', '左屈', '除外', '先天性側彎'])
rows = []
filtered_shape_lines = []
for shape_line in shape_lines:
    ap = 'ap_whole' in shape_line.content.flags
    if not ap:
        continue
    if any(flag in shape_line.content.flags for flag in ban_flags):
        continue
    filename = shape_line.filename
    post_op = '術後' in shape_line.content.flags
    rows.append((filename, shape_line.content.imagePath, post_op))
    filtered_shape_lines.append(shape_line)

df = pd.DataFrame(rows, columns=['filename', 'image_path', 'post_op'])
df.head()
df.describe().T
# %%
shape_lines[2].content.flags
# %%
from szkspyne.coord import Spine
first = shape_lines[0].content
first_spine = Spine.from_shape_dict(first)

params = []
for line in shape_lines:
    shape_dict = line.content
    try:
        spine = Spine.from_shape_dict(shape_dict)
    except ValueError as e:
        print(f"Error processing {shape_dict.filename}: {e}")
        continue
    tf = spine.estimate_transform(first_spine, spine)
    params.append((tf.scale, tf.translation[0], tf.translation[1], tf.rotation))
df_params = pd.DataFrame(params, columns=['scale', 'translation_x', 'translation_y', 'rotation'])
# df_params.index = [line.filename for line in shape_lines]
df_params
# %%
# calculate mahalanobis distance
from scipy.spatial import distance
import numpy as np
mean = df_params.mean()
cov = df_params.cov()
inv_cov = np.linalg.inv(cov)
print("Mean:\n", mean)
print("Covariance:\n", cov)
mahalanobis_distances = []
for _, row in df_params.iterrows():
    diff = row - mean
    mahalanobis_distance = distance.mahalanobis(diff, np.zeros_like(diff), inv_cov)
    mahalanobis_distances.append(mahalanobis_distance)
df_params['mahalanobis_distance'] = mahalanobis_distances
df_params.sort_values('mahalanobis_distance', ascending=True, inplace=True)
df_params.head(20)

# %%
from pathlib import Path
reference_index = df_params.index[0]
reference_spine = Spine.from_shape_dict(shape_lines[reference_index].content)

image_outdir.mkdir(exist_ok=True, parents=True)

with open(image_outdir/'reference_spine.txt', 'w') as f:
    f.write(shape_lines[reference_index].filename + '\n')

# %%
from PIL import Image
from skimage.transform import rescale
from tqdm import tqdm
import joblib
import pydicom
import matplotlib.pyplot as plt
from utils import load_image, save_as_img, to_u8_range

# image_outdir = Path('../../../data/generative_dataset') / 'cyclegan_prepost'
image_outdir.mkdir(exist_ok=True, parents=True)


ref_shape = shape_lines[reference_index].content
reference_original_image = load_image(ref_shape.imagePath)
ref_spine = Spine.from_shape_dict(shape_lines[reference_index].content)
ref_spine_align = Spine.ReferenceSpine(ref_spine, reference_original_image)
ref_cropped_spine, reference_image  = ref_spine_align.crop_self()

plot_result = False
if plot_result:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Load the reference image
    ax1.imshow(reference_original_image, cmap='gray')
    ax1.plot(ref_spine.vertebrae[:, :, 0].ravel(), ref_spine.vertebrae[:, :, 1].ravel(), 'o', markersize=3, color='blue')
    # Crop the reference spine
    ax2.imshow(reference_image, cmap='gray')
    ax2.plot(ref_cropped_spine[:, :, 0].ravel(), ref_cropped_spine[:, :, 1].ravel(), 'o', markersize=3, color='red')
    plt.show()
    print('crop param', ref_spine_align.crop_params)
    print('image shapes', reference_image.shape, reference_original_image.shape)
#%%

# ref_cropped_image,  = ref_spine.crop_image(reference_image, scale_factor=scale_factor)


list_filename_shape_line = []

for spine_line in filtered_shape_lines:
    shape_dict = spine_line.content
    post_op = '術後' in shape_dict.flags 
    category = 'B' if post_op else 'A'
    train_or_test = 'test' if 'MSJ' in shape_dict.imagePath else 'train'
    subdir = f'{train_or_test}{category}'
    file_stem = Path(shape_dict.imagePath).stem
    filename = image_outdir / subdir / f'{file_stem}.jpg'
    filename.parent.mkdir(exist_ok=True, parents=True)
    list_filename_shape_line.append((filename, spine_line))


def align_spine(filename, shape_line):
    shape_dict = shape_line.content
    try:
        spine = Spine.from_shape_dict(shape_dict)
    except ValueError as e:
        print(f"Error processing {shape_dict.filename}: {e}")
        return
    

    # Load the image
    apply_clahe = False
    if 'Dairo' in shape_dict.imagePath:
        apply_clahe = True
    image = load_image(shape_dict.imagePath)
    if plot_result:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image, cmap='gray')
        ax1.plot(spine.vertebrae[:, :, 0].ravel(), spine.vertebrae[:, :, 1].ravel(), 'o', markersize=3, color='blue')

    # Crop the image and spine
    cropped_spine, cropped_image, cropped_shape_dict = ref_spine_align.align(spine, image)
    if plot_result:
        print(cropped_image.shape)
        ax2.imshow(cropped_image, cmap='gray')
        ax2.plot(cropped_spine[:, :, 0].ravel(), cropped_spine[:, :, 1].ravel(), 'o', markersize=3, color='red')
        ax2.plot(ref_cropped_spine[:, :, 0].ravel(), ref_cropped_spine[:, :, 1].ravel(), 'o', markersize=3, color='green')
        plt.show()

    # rescale the cropped image and spine
    scale_factor = 0.5
    rescaled_cropped_image = rescale(cropped_image, scale_factor, anti_aliasing=True)
    cropped_shape_dict = cropped_shape_dict.transform(lambda points: points * scale_factor)
    img_u8 = to_u8_range(rescaled_cropped_image, percentile=1, clahe=apply_clahe)

    save_as_img(filename, img_u8)
    cropped_shape_line = labelme_utils.ShapeDictLine(
        filename=shape_line.filename,
        content=cropped_shape_dict
    )
    return ((filename.relative_to(image_outdir),shape_dict.imagePath), cropped_shape_line)

result = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(align_spine)(filename, shape_line) for filename, shape_line in tqdm(list_filename_shape_line)
)

path_pairs, shape_dicts= zip(*[r for r in result if r is not None])
shape_dicts: List[labelme_utils.ShapeDictLine]  = shape_dicts

with open(image_outdir / 'path_pairs.txt', 'w') as f:
    for pre_path, post_path in path_pairs:
        f.write(f'{pre_path},{post_path}\n')

lm_lines = [sd.into_labelme_line() for sd in shape_dicts]
labelme_utils.LabelMeLine.save_to_ndjson(lm_lines, image_outdir / 'keypoints.ndjson')
    
# %%
