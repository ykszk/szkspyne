#%%
from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class ActiveShapeModel:
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_scaled_components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    template_points: np.ndarray
    template_reference: List[Optional[float]]
    labels: List[str]
    point_counts: List[int]

    def __init__(self, **kwargs):
        self.pca_mean = kwargs['mean']
        self.pca_components = kwargs['components']
        self.pca_scaled_components = kwargs['components'] * np.sqrt(kwargs['explained_variance'])[:, np.newaxis]
        self.explained_variance = kwargs['explained_variance']
        self.explained_variance_ratio = kwargs['explained_variance_ratio']
        self.template_points = kwargs['template_points']
        self.template_reference = kwargs['template_reference']
        self.labels = kwargs['labels']
        self.point_counts = kwargs['point_counts']
        self.split = [0] + np.cumsum(2 * np.array(self.point_counts)).tolist()

    @staticmethod
    def load(filename: str) -> 'ActiveShapeModel':
        """
        Load the ActiveShapeModel from a file.
        The file should contain the PCA components and the template points.
        """
        data = np.load(filename, allow_pickle=True)
        return ActiveShapeModel(
            pca_mean=data['mean'],
            pca_components=data['components'],
            explained_variance=data['explained_variance'],
            explained_variance_ratio=data['explained_variance_ratio'],
            template_points=data['template_points'],
            template_reference=data['template_reference'].tolist(),
            labels=data['labels'].tolist(),
            point_counts=data['point_counts'].tolist()
        )
    
    def unravel(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Unravel the points from the PCA space to the original space.
        """
        unraveled = []
        for i in range(1, len(self.split)):
            start, end = self.split[i - 1], self.split[i]
            unraveled.append(points[start:end].reshape(-1, 2))
        return unraveled
    
    def inverse_transform(self, deformation: np.ndarray) -> np.ndarray:
        """
        Deform the template points using the PCA deformation.
        """
        if deformation.shape[0] != self.pca_components.shape[0]:
            raise ValueError("Deformation must match the number of PCA components.")
        tfed = self.pca_mean + deformation @ self.pca_scaled_components
        return tfed
    
    def deform(self, deformation: np.ndarray) -> List[np.ndarray]:
        return self.unravel(self.inverse_transform(deformation))
    
    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Transform the points to the PCA space.
        """
        if points.shape[0] != self.pca_mean.shape[0]:
            raise ValueError("Points must match the number of PCA mean dimensions.")
        flat_points = points.flatten()
        return (flat_points - self.pca_mean) @ self.pca_components.T / np.sqrt(self.explained_variance)
    
    def undo_deform(self, deformed_points: List[np.ndarray]) -> np.ndarray:
        """
        Undo the deformation to get back to the original points.
        """
        cat = np.concatenate(deformed_points, axis=0)
        return self.transform(cat.ravel())
    
model_filename = '/tmp/tmp_neck.npz'

asm = ActiveShapeModel.load(model_filename)
mean_point_sets = asm.unravel(asm.pca_mean)
import matplotlib.pyplot as plt

def plot_points(points: List[np.ndarray], labels: List[str], title: str = 'Active Shape Model'):
    plt.figure(figsize=(10, 6))
    for pts, label in zip(points, labels):
        plt.scatter(pts[:, 0], pts[:, 1], label=label)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

plot_points(mean_point_sets, asm.labels, 'Active Shape Model Points')

deformation = np.zeros((asm.pca_components.shape[0],))
for first_component in [-2, -1, 0, 1, 2]:
    deformation[0] = first_component  # Example deformation
    deformed_points = asm.deform(deformation)
    plot_points(deformed_points, asm.labels, 'Deformed Active Shape Model Points')
    params = asm.undo_deform(deformed_points)

# %%

# %%
import szkspyne.settings as settings
settings.ASM_LABEL_NAMES = ['TL', 'TR', 'BL', 'BR', "Lamina", "AnteriorC1Arch", "AnteriorDens", "PosteriorDens"]
settings.ANCHOR_LABEL_NAMES = ["AnteriorDens", "PosteriorDens", "Lamina1", "C3_BL", "C3_BR"]

ndjson_filename = '/tmp/tmp_neck.ndjson'
from szkspyne import labelme_utils
from szkspyne.asm.point_data import SpinePointData, MovablePoints
data = labelme_utils.ShapeDictLine.from_ndjson(ndjson_filename)
print(f'Loaded {len(data)} lines from {ndjson_filename}')
import matplotlib.pyplot as plt
# %%
# spine_data = [SpinePointData.from_shape_dict(d.content) for d in data]
spine_data = []
filtered_data = []
for i in range(len(data)):
    line = data[i]
    # print(line.filename)
    try:
        spine_data.append(SpinePointData.from_shape_dict(line.content))
        filtered_data.append(line)
    except Exception as e:
        print(f'Error processing line {i}: {e}')
        continue
print(f'Loaded {len(spine_data)} spine data points')
# %%
template = MovablePoints(asm.template_points, asm.template_reference, asm.point_counts)
# %%
aligned_data = [template.align(d)[0] for d in spine_data]

#%%
params = [asm.transform(d.ravel()) for d in aligned_data]
# %%
# %%
from PIL import Image
from skimage.transform import warp
from tqdm import tqdm

def ticks_off():
    plt.xticks([])
    plt.yticks([])
    [s.set_visible(False) for s in plt.gca().spines.values()]

#%%
from functools import lru_cache
@lru_cache(maxsize=None)
def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)

param_index = 2
param_index_label = f'PC{param_index + 1}'
frames = []
i_frame = 0
fps = 24
max_value = 2
sequence = np.concatenate([np.linspace(0, -max_value, 24*3), np.linspace(-max_value, 0, 24*3), np.linspace(0, max_value, 24*3), np.linspace(max_value,0,24*3)], axis=0)
for param_value in tqdm(sequence):
    plt.clf()
    target_params = np.zeros((asm.pca_components.shape[0],))
    target_params[param_index] = param_value  # Example deformation
    l2_norms = [np.linalg.norm((p - target_params)[:10]) for p in params]
    # l2_norms = [abs(p[param_index] - param_value) for p in params]
    closest_index = np.argmin(l2_norms)


    image_path = filtered_data[closest_index].content.imagePath
    # image = Image.open(image_path)
    image = load_image(image_path)
    tf = template.calculate_transform(spine_data[closest_index])
    warped_image = warp(np.array(image), tf.inverse)
    plt.imshow(warped_image, 'gray')
    point_sets = asm.deform(target_params)
    # point_sets = asm.unravel(aligned_data[closest_index].ravel())
    for points, label in zip(point_sets, asm.labels):
        plt.scatter(points[:, 0], points[:, 1], s=10, label=label)
    
    plt.title(f'{param_index_label} = {param_value:.2f}')
    ticks_off()

    plt.gcf().savefig(f'/tmp/frames/{i_frame:04d}.png', format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    i_frame += 1

#%%
'ffmpeg -y -framerate 24 -i /tmp/frames/%04d.png -c:v libx264 -vf format=yuv420p -r 24  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" /tmp/output.mp4 && open /tmp/output.mp4'

# %%
# worst_param_index = np.argmax([np.linalg.norm(p) for p in params])
arg_sorted = np.argsort([np.linalg.norm(p) for p in params])
worst_param_index = arg_sorted[-10]  # Get the index of the worst parameter
worst_param_value = params[worst_param_index][param_index]
print(f'Worst parameter index: {worst_param_index}, value: {worst_param_value}')
worst_image_path = filtered_data[worst_param_index].content.imagePath
worst_image = Image.open(worst_image_path)
plt.imshow(worst_image, 'gray')
ticks_off()
# %%
params[worst_param_index]