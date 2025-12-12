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
            mean=data['mean'],
            components=data['components'],
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

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# command line arguments
parser = argparse.ArgumentParser(description="Visualize Active Shape Model")
parser.add_argument('model', type=Path, help="Path to the ASM model file")
parser.add_argument('--pc', type=int, default=0, help="Principal component index to visualize")
parser.add_argument('--pc_start', type=float, default=-2.0, help="Start value for the selected principal component")
parser.add_argument('--pc_end', type=float, default=2.0, help="End value for the selected principal component")
parser.add_argument('--pc_num', type=int, default=100, help="Number of values for the selected principal component to visualize if pc_values is not given")
parser.add_argument('--fps', type=int, default=10, help="Frames per second for animation")
parser.add_argument('--interval', type=int, default=100, help="Interval between frames in milliseconds")
# save figure of cumulative explained variance
parser.add_argument('--variance_fig', type=Path, help="Path to save the cumulative explained variance figure")
parser.add_argument('output', type=Path, help="Output MP4 filename")
args = parser.parse_args()

model_filename = args.model

asm = ActiveShapeModel.load(model_filename)


if args.variance_fig:
    cum_variance = np.cumsum(asm.explained_variance_ratio)
    plt.clf()
    plt.plot(np.arange(1, len(cum_variance) + 1), cum_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.grid()
    print(f'Saving cumulative explained variance figure to {args.variance_fig}...')
    plt.savefig(args.variance_fig, dpi=200)


def plot_points(points: List[np.ndarray], labels: List[str], title: str = 'Active Shape Model'):
    plt.clf()
    for pts, label in zip(points, labels):
        plt.scatter(pts[:, 0], pts[:, 1], label=label)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.axis('equal')
    plt.grid()

pc_values = np.linspace(args.pc_start, args.pc_end, args.pc_num)

def plot_frame(frame):
    deformation = np.zeros((asm.pca_components.shape[0],))
    pc_value = pc_values[frame]
    deformation[args.pc] = pc_value
    deformed_points = asm.deform(deformation)
    plot_points(deformed_points, asm.labels, f'Deformed Active Shape Model Points (PC{args.pc + 1} = {pc_value:.2f})')

fig = plt.figure(figsize=(6, 6), dpi=200)
ani = animation.FuncAnimation(fig, plot_frame, frames=len(pc_values), interval=args.interval)
print(f'Saving animation to {args.output}...')
ani.save(args.output, writer='ffmpeg', fps=args.fps)
