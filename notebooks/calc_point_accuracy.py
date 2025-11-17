#%%
from pathlib import Path
import numpy as np
import pandas as pd

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
# %%
import matplotlib.pyplot as plt
import pydicom
d_ai = lines_to_dict(ai_points)
d_ref = lines_to_dict(ref_points_lines)
tab10_colors = plt.get_cmap('tab10').colors

if image_type == 'ap_whole':
    reps = ["Dairo^A13^07795786_20140526_CR_SA13_20231106183222000009dc00088396_C1E18985-BBD6-4150-BB32-E983FFA148FF","Dairo^A13^08638374_20220711_CR_SA13_20231106185843000009dc00088472_B661A42A-D62D-4798-9450-4FD169BA655B"]
else:
    reps = ["OSC00511_20210804_2","SNDAISY0859-RN_6MO_0001^Patient_000002_DX_PROFIL"]
# for name in d_ai.keys():
for name in reps:
    coronal_ai = d_ai[name]
    coronal_ref = d_ref[name]
    image_filename = Path(coronal_ai.image_metadata.path)
    ds = pydicom.dcmread(image_filename)
    img = ds.pixel_array
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        img = np.max(img) - img
    spine_ai = np.array(coronal_ai.spine)
    spine_ref = np.array(coronal_ref.spine)
    # crop image using points with some margin
    margin = 100
    x_min = int(min(np.min(spine_ai[:, :, 0]), np.min(spine_ref[:, :, 0]))) - margin
    x_max = int(max(np.max(spine_ai[:, :, 0]), np.max(spine_ref[:, :, 0]))) + margin
    y_min = int(min(np.min(spine_ai[:, :, 1]), np.min(spine_ref[:, :, 1]))) - margin
    y_max = int(max(np.max(spine_ai[:, :, 1]), np.max(spine_ref[:, :, 1]))) + margin
    img = img[y_min:y_max, x_min:x_max]
    spine_ai[:, :, 0] -= x_min
    spine_ai[:, :, 1] -= y_min
    spine_ref[:, :, 0] -= x_min
    spine_ref[:, :, 1] -= y_min

    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(img, cmap='gray')
    for i in range(4):
        xy_ai = spine_ai[:, i]
        xy_ref = spine_ref[:, i]
        plt.scatter(xy_ref[:, 0], xy_ref[:, 1], color=tab10_colors[i], alpha=0.3)
        plt.scatter(xy_ai[:, 0], xy_ai[:, 1], marker='x', color=tab10_colors[i], alpha=0.7)
    # rotate plt for better view
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()
#%%

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
    df_diff['distance'] = np.linalg.norm(diffs, axis=1)
    df_diff['name'] = name
    df_diff['point_index'] = np.tile(np.arange(4), 17)
    df_diff['level'] = np.repeat(np.arange(17), 4)
    all_diffs.append(df_diff)
    # break

# for cp_line in ai_coronal_points_lines:
#     coronal_points = cp_line.content
#     spine = np.array(coronal_points.spine)
#     for i in range(4):
#         xy = spine[:, i]
#         plt.scatter(xy[:, 0], xy[:, 1])
#     break
df_diffs = pd.concat(all_diffs, axis=0)
df_diffs.to_csv(result_dir / 'postprocess/spine_keypoint_differences.csv', index=False)
# %%

plt.figure(figsize=(6, 6), dpi=100)
# point_names = ['TL', 'TR', 'BL', 'BR']
point_names = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']
units = 'mm'
for point_index, point_name in enumerate(point_names):
    df_subset = df_diffs[df_diffs['point_index'] == point_index]
    l2_norm = np.linalg.norm(df_subset[['x', 'y']], axis=1)
    mean_error = np.mean(l2_norm)
    std = np.std(l2_norm)
    print(f'Point {point_name}: Mean Error = {mean_error:.2f} {units}, Std = {std:.2f} {units}')

    plt.subplot(2, 2, point_index + 1)
    plt.scatter(df_subset['x'], df_subset['y'], alpha=0.1, label=f'Point {point_name}')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='red', linestyle='--', linewidth=0.5)
    plt.title(point_name)
    plt.gca().invert_yaxis()
    plt.xlabel(f'X [{units}]')
    plt.ylabel(f'Y [{units}]')
    plt.axis('equal')
    plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# scatter plot level per point_name
plt.figure(figsize=(6, 6 ), dpi=100)
level_names = ['C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
for level, level_name in enumerate(level_names):
    df_subset = df_diffs[df_diffs['level'] == level]
    for point_index, point_name in enumerate(point_names):
        df_point = df_subset[df_subset['point_index'] == point_index]
        l2_norm = np.linalg.norm(df_point[['x', 'y']], axis=1)
        mean_error = np.mean(l2_norm)
        std = np.std(l2_norm)
        plt.subplot(2, 2, point_index + 1)
        plt.scatter(df_point['x'], df_point['y'], alpha=0.1, label=f'Level {level_name}')
        plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='red', linestyle='--', linewidth=0.5)
        plt.title(f'{point_name} ($E[||\\vec{{e}}||_2]$={mean_error:.2f}, std={std:.2f})')
        plt.gca().invert_yaxis()
        plt.xlabel(f'Error in X [{units}]')
        plt.ylabel(f'Error in Y [{units}]')
        plt.axis('equal')
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle(level_name, y=1.02)
    plt.show()
    break

# %%
