#%%
from typing import List, Optional

from skimage.transform import SimilarityTransform, warp
import numpy as np
from logzero import logger
from szkspyne import settings, labelme_utils
from dataclasses import dataclass

#%%

@dataclass
class MovablePoints():
    points: np.ndarray
    reference_points: List[Optional[float]]
    point_counts: List[int]

    def __init__(self, points: np.ndarray, reference_points: List[Optional[float]], point_counts: List[int]):
        self.points = points
        self.reference_points = reference_points
        self.point_counts = point_counts

    def calculate_transform(self, src_points: 'MovablePoints') -> SimilarityTransform:
        """
        Calculate the transform parameters to transform the src_points to the template points.
        The transform parameters are defined as (scale, rotation, translation).
        """
        src = src_points.reference_points
        dst = self.reference_points
        # calculate similarity transform parameters
        tf = SimilarityTransform()
        tf.estimate(src, dst)
        return tf
    
    def align(self, src_points: 'MovablePoints') -> tuple:
        """
        Align the src_points to the template points using the transform parameters.
        The transform parameters are defined as (scale, rotation, translation).
        """
        tf = self.calculate_transform(src_points)
        transformed_points = tf(src_points.points)
        return transformed_points, tf
    
    @staticmethod
    def select_template(mps: List['MovablePoints']) -> 'MovablePoints':
        """
        Create a TemplatePoints from a list of MovablePoints.
        MovablePoints that is the closest to the mean of the transformation parameters is selected as the template.
        """
        if not mps:
            raise ValueError("List of MovablePoints is empty")
        tentative_template = mps[0]
        if len(mps) == 1:
            return tentative_template
        tfs = [tentative_template.calculate_transform(mp) for mp in mps[1:]]
        param_arrs = np.array([[tf.scale, tf.translation[0], tf.translation[1], tf.rotation] for tf in tfs])
        mean_params = np.mean(param_arrs, axis=0)
        # calculate the mahalanobis distance from the mean
        cov = np.cov(param_arrs.T)
        inv_cov = np.linalg.inv(cov)
        distances = []
        for tf in tfs:
            diff = np.array([tf.scale, tf.translation[0], tf.translation[1], tf.rotation]) - mean_params
            mahalanobis_distance = np.sqrt(diff @ inv_cov @ diff.T)
            distances.append(mahalanobis_distance)
        # find the index of the minimum distance
        min_index = np.argmin(distances)
        return mps[min_index]

    @staticmethod
    def from_labels(shape_dict: labelme_utils.ShapeDict, point_labels: List[str], reference_labels: List[str]) -> 'MovablePoints':
        '''
        Convert LabelMe data to SpinePointData
        '''
        points = []
        point_counts = []
        for label in point_labels:
            pts = shape_dict.shapes['point'][label]
            points.extend(pts)
            point_counts.append(len(pts))
        points = np.array(points, dtype=np.float32)
        points = points[:, 0] # squeeze (N, 1, 2) -> (N, 2)

        reference_points = []
        for label in reference_labels:
            if label in shape_dict.shapes['point']:
                pts = shape_dict.shapes['point'][label]
                if len(pts) > 1:
                    logger.warning(
                        'More than one reference point for %s: %s. Using the first one.', label, pts
                    )
                reference_points.append(pts[0][0])
            else:
                reference_points.append(None)
        return MovablePoints(points=points, reference_points=reference_points, point_counts=point_counts)

class SpinePointData(MovablePoints):
    @staticmethod
    def from_shape_dict(shape_dict: labelme_utils.ShapeDict) -> 'MovablePoints':
        """
        Convert LabelMe data to SpinePointData
        """
        return MovablePoints.from_labels(
            shape_dict,
            settings.ASM_LABEL_NAMES,
            settings.ANCHOR_LABEL_NAMES
        )


# %%
from sklearn.decomposition import PCA

@dataclass
class ActiveShapeModelBuilder:
    pca: PCA
    template: MovablePoints

    @staticmethod
    def from_data(data: List[MovablePoints]) -> 'ActiveShapeModelBuilder':
        """
        Create an ActiveShapeModel from a list of MovablePoints.
        The model is created by selecting the template from the data.
        """
        template = MovablePoints.select_template(data)
        aligned_data = [template.align(d)[0] for d in data]
        array = np.array(aligned_data, dtype=np.float64)
        array = array.reshape(array.shape[0], -1)
        pca = PCA()
        pca.fit(array)

        return ActiveShapeModelBuilder(pca=pca, template=template)
    
    def save(self, filename: str, labels: List[str], point_counts: List[int]) -> None:
        """
        Save the ActiveShapeModel to a file.
        The file will contain the PCA components and the template points.
        """
        np.savez(filename, pca_mean=self.pca.mean_, pca_components=self.pca.components_, explained_variance=self.pca.explained_variance_, explained_variance_ratio=self.pca.explained_variance_ratio_, template_points=self.template.points, template_reference=self.template.reference_points, labels=labels, point_counts=point_counts)


def main():
    import sys
    import argparse
    from szkspyne import labelme_utils

    parser = argparse.ArgumentParser(
        description='Create an Active Shape Model from spine point data.')
    parser.add_argument('output', help='Output npz filename')

    args = parser.parse_args()

    shape_dicts = labelme_utils.ShapeDictLine.from_lines(sys.stdin)
    logger.info('Loaded %d shape dicts', len(shape_dicts))
    spine_data = [SpinePointData.from_shape_dict(shape_dict.content) for shape_dict in shape_dicts]
    logger.info('Point counts: %s', spine_data[0].point_counts)

    logger.info('Building Active Shape Model from %d spine data', len(spine_data))
    asm = ActiveShapeModelBuilder.from_data(spine_data)
    # %%
    asm.save(args.output, settings.ASM_LABEL_NAMES, spine_data[0].point_counts)

if __name__ == '__main__':
    main()