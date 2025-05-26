#%%
from szkspyne import labelme_utils
import json

ndjson_filename = '../../script/group_id_dicom_manual_annotation.ndjson'

with open(ndjson_filename, 'r') as f:
    lines = f.readlines()
objects = [json.loads(line) for line in lines]
shape_lines = [labelme_utils.ShapeDictLine.model_validate(obj) for obj in objects]
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
reference_index = df_params.index[0]
reference_spine = Spine.from_shape_dict(shape_lines[reference_index].content)
# %%
from pathlib import Path
from PIL import Image
from skimage.transform import rescale
from tqdm import tqdm
import joblib
import pydicom
import matplotlib.pyplot as plt

image_outdir = Path('../../../data/generative_dataset') / 'cyclegan_postpre'
image_outdir.mkdir(exist_ok=True, parents=True)

def load_image(image_path, percentile=None, clahe=False):
    dcm = pydicom.dcmread(image_path)
    # invert pixel array if necessary
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.invert(dcm.pixel_array)
    else:
        pixel_array = dcm.pixel_array
    # convert to uint8
    # dcm.pixel_array = dcm.pixel_array.astype(np.uint8)
    if clahe:
        from skimage import exposure
        pixel_array = exposure.equalize_adapthist(pixel_array, clip_limit=0.01)
        pixel_array = (pixel_array * 255).astype(np.uint8)
    if percentile is not None:
        return normalize(pixel_array, percentile=percentile)
    return pixel_array
# img1 = pydicom.dcmread(shape_dict1.imagePath).pixel_array
# img2 = pydicom.dcmread(shape_dict2.imagePath).pixel_array

def normalize(x, wlww=None, minmax=None, percentile=None):
    if wlww is None and minmax is None and percentile is None:
        raise RuntimeError('Either wlww or minmax is required')

    if wlww is not None:
        wl, ww = wlww
        minmax = (wl-ww/2, wl+ww/2)

    if percentile is not None:
        minmax = np.percentile(x,[percentile,100-percentile])

    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255*(x - minmax[0])/(minmax[1]-minmax[0])).astype(np.uint8)

ref_shape = shape_lines[reference_index].content
reference_original_image = load_image(ref_shape.imagePath, percentile=4)
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


from collections import defaultdict
counters = defaultdict(int)

list_filename_shape_line = []

for spine_line in filtered_shape_lines:
    shape_dict = spine_line.content
    post_op = '術後' in shape_dict.flags 
    subdir = 'trainB' if post_op else 'trainA'
    count = counters[subdir]
    counters[subdir] += 1
    filename = image_outdir / subdir / f'{count:05d}.jpg'
    filename.parent.mkdir(exist_ok=True, parents=True)
    list_filename_shape_line.append((filename, spine_line))

def save_as_img(filename, image):
    # convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(filename)

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
    image = load_image(shape_dict.imagePath, percentile=4, clahe=apply_clahe)
    if plot_result:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image, cmap='gray')
        ax1.plot(spine.vertebrae[:, :, 0].ravel(), spine.vertebrae[:, :, 1].ravel(), 'o', markersize=3, color='blue')

    # Crop the image and spine
    cropped_spine, cropped_image = ref_spine_align.align(spine, image)
    if plot_result:
        print(cropped_image.shape)
        ax2.imshow(cropped_image, cmap='gray')
        ax2.plot(cropped_spine[:, :, 0].ravel(), cropped_spine[:, :, 1].ravel(), 'o', markersize=3, color='red')
        ax2.plot(ref_cropped_spine[:, :, 0].ravel(), ref_cropped_spine[:, :, 1].ravel(), 'o', markersize=3, color='green')
        plt.show()

    # rescale the cropped image and spine
    scale_factor = 0.5
    rescaled_cropped_image = rescale(cropped_image, scale_factor, anti_aliasing=True)

    save_as_img(filename, rescaled_cropped_image)
    return (filename.relative_to(image_outdir),shape_dict.imagePath)

path_pairs = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(align_spine)(filename, shape_line) for filename, shape_line in tqdm(list_filename_shape_line)
)

with open(image_outdir / 'path_pairs.txt', 'w') as f:
    for pre_path, post_path in path_pairs:
        f.write(f'{pre_path},{post_path}\n')
    
# %%
