#%%
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom

from szkspyne.utils.scolrs import SpinalPointsLine, SpinalPoints

# %%
result_dir = Path("/poseidon/users/suzuki/projects/scoliosis/result/exp20250109/")
image_type = 'ap_whole'
# image_type = 'lateral_whole'
ndjson_ai = result_dir / f'auto/keypoints/{image_type}/no-bend_final_keypoints_native.ndjson'
ndjson_ref = result_dir / f'reference/keypoints/{image_type}/no-bend_final_keypoints_native.ndjson'

ai_points: list[SpinalPointsLine] = SpinalPointsLine.from_ndjson(ndjson_ai)
ref_points_lines: list[SpinalPointsLine] = SpinalPointsLine.from_ndjson(ndjson_ref)
assert len(ref_points_lines) == len(ai_points)
len(ai_points)

def lines_to_dict(lines: list[SpinalPointsLine]) -> dict[str, SpinalPoints]:
    return {Path(line.filename).stem: line.content for line in lines}

d_ai = lines_to_dict(ai_points)
d_ref = lines_to_dict(ref_points_lines)

# %%

all_diffs = []
for name in d_ai.keys():
    coronal_ai = d_ai[name]
    coronal_ref = d_ref[name]
    spacings = coronal_ai.image_metadata.spacing_xy
    spine_ai = np.array(coronal_ai.spine)[:17] * spacings
    spine_ref = np.array(coronal_ref.spine)[:17] * spacings
    diffs = spine_ai - spine_ref
    diffs = diffs.reshape(-1, 2)
    # plt.scatter(diffs[:, 0], diffs[:, 1], alpha=0.1)
    df_diff = pd.DataFrame(diffs, columns=['x', 'y'])
    df_diff['name'] = name
    df_diff['point_index'] = np.tile(np.arange(4), 17)
    df_diff['level'] = np.repeat(np.arange(17), 4)
    all_diffs.append(df_diff)
df_diffs = pd.concat(all_diffs, axis=0)

# %%
import matplotlib.pyplot as plt

rep_name = d_ai.keys().__iter__().__next__()
spinal_points = d_ai[rep_name]
image_filename = Path(spinal_points.image_metadata.path)
ds = pydicom.dcmread(image_filename)
img = ds.pixel_array
if ds.PhotometricInterpretation == 'MONOCHROME1':
    img = np.max(img) - img

plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(img, cmap='gray')

#%%
t1 = np.array(spinal_points.spine[1])
# crop image using t1 with some margin
margin = 40
x_min = int(min(t1[:, 0])) - margin
x_max = int(max(t1[:, 0])) + margin
y_min = int(min(t1[:, 1])) - margin
y_max = int(max(t1[:, 1])) + margin
img_cropped = img[y_min:y_max, x_min:x_max]
spacings = spinal_points.image_metadata.spacing_xy
plt.figure(figsize=(6, 6), dpi=200)
extent = [0, (x_max - x_min) * spacings[0], (y_max - y_min) * spacings[1], 0]
plt.imshow(img_cropped, cmap='gray', extent=extent)
point_names = ['TL', 'TR', 'BL', 'BR']
for i, point_name in enumerate(point_names):
    t1_xy = t1[i] * spacings
    diffs = df_diffs[df_diffs['point_index'] == i]
    clip = 30
    clipped_diffs = np.clip(diffs[['x', 'y']].values, -clip, clip)
    centered_diffs = clipped_diffs + t1_xy - np.array([x_min * spacings[0], y_min * spacings[1]])
    plt.scatter(centered_diffs[:, 0], centered_diffs[:, 1], alpha=0.1, label=point_name, s=10)
plt.legend()
plt.xlabel('mm')
plt.ylabel('mm')
plt.grid(linestyle='dashed', color='gray', alpha=0.3)
plt.gca().set_aspect('equal', adjustable='box')

# %%
# calculate average width and height
all_widths, all_heights = [], []
for name in d_ai.keys():
    ref = d_ref[name]
    spacings = ref.image_metadata.spacing_xy
    spine = np.array(ref.spine)[:17] * spacings
    # width = l2_norm of (TR - TL) and (BR - BL)
    widths = np.linalg.norm(spine[:, 1] - spine[:, 0], axis=1)
    all_widths.append(widths)
    width = np.linalg.norm(spine[:, 3] - spine[:, 2], axis=1)
    all_widths.append(width)
    # height = l2_norm of (TL - BL) and (TR - BR)
    heights = np.linalg.norm(spine[:, 0] - spine[:, 2], axis=1)
    all_heights.append(heights)
    heights = np.linalg.norm(spine[:, 1] - spine[:, 3], axis=1)
    all_heights.append(heights)
avg_width = np.mean(np.concatenate(all_widths))
std_width = np.std(np.concatenate(all_widths))
avg_height = np.mean(np.concatenate(all_heights))
std_height = np.std(np.concatenate(all_heights))
print(f'Average width: {avg_width:.2f} mm (std: {std_width:.2f} mm)')
print(f'Average height: {avg_height:.2f} mm (std: {std_height:.2f} mm)')

# %%
