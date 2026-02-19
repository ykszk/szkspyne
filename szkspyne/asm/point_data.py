#%%
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from logzero import logger
from skimage.transform import SimilarityTransform

from szkspyne import labelme_utils, settings

#%%

@dataclass
class MovablePoints():
    points: np.ndarray
    reference_points: List[Optional[float]]
    ''' Points for calculation of the transform. Potentially None if the point is missing. '''
    point_counts: List[int]

    def __init__(self, points: np.ndarray, reference_points: List[Optional[float]], point_counts: List[int]):
        self.points = points
        self.reference_points = reference_points
        if any(rp is None for rp in reference_points):
            logger.warning(
                'Some reference points are None: %s', reference_points
            )
        self.point_counts = point_counts

    def calculate_transform(self, src_points: 'MovablePoints', inverse=False) -> SimilarityTransform:
        """
        Calculate the transform parameters to transform the src_points to the template points.
        The transform parameters are defined as (scale, rotation, translation).
        """
        src = src_points.reference_points
        dst = self.reference_points
        # calculate similarity transform parameters
        tf = SimilarityTransform()
        if inverse:
            tf.estimate(dst, src)
        else:
            tf.estimate(src, dst)
        return tf
    
    def calculate_transform_with_missing(self, src_points: 'MovablePoints', inverse=False) -> SimilarityTransform:
        """
        Calculate the transform parameters to transform the src_points to the template points.
        The transform parameters are defined as (scale, rotation, translation).
        This method handles missing reference points by ignoring them in the calculation.
        """
        valid_indices = [i for i, p in enumerate(src_points.reference_points) if p is not None]
        src = np.array([src_points.reference_points[i] for i in valid_indices])
        dst = np.array([self.reference_points[i] for i in valid_indices])
        if len(src) < 2 or len(dst) < 2:
            raise ValueError("Not enough reference points to calculate transform")
        tf = SimilarityTransform()
        if inverse:
            tf.estimate(dst, src)
        else:
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
        reference_points = mps[min_index].reference_points
        if any(rp is None for rp in reference_points):
            logger.warning(
                'Some reference points are None in the selected template.'
            )
        return mps[min_index]

    @staticmethod
    def from_labels(shape_dict: labelme_utils.ShapeDict, point_labels: List[str], reference_labels: List[str]) -> 'MovablePoints':
        '''
        Convert LabelMe data to SpinePointData
        '''
        points = []
        point_counts = []
        for label in point_labels:
            if label in shape_dict.shapes['point']:
                pts = shape_dict.shapes['point'][label]
            else:
                logger.warning('No points found for label %s', label)
                pts = np.empty((0, 2), dtype=np.float32)
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

class PresetPointData(MovablePoints):
    @staticmethod
    def from_shape_dict(shape_dict: labelme_utils.ShapeDict) -> 'MovablePoints':
        """
        Convert LabelMe data to PresetPointData based on settings.ASM_LABEL_NAMES and settings.ANCHOR_LABEL_NAMES.
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
    
    def save(self, filename: str, model_name: str, labels: List[str], reference_labels: List[str]) -> None:
        """
        Save the ActiveShapeModel to a file.
        The file will contain the PCA components and the template points.
        """
        np.savez(filename, model_name=model_name, mean=self.pca.mean_, components=self.pca.components_, explained_variance=self.pca.explained_variance_, explained_variance_ratio=self.pca.explained_variance_ratio_, template_points=self.template.points, template_reference=self.template.reference_points, labels=labels, point_counts=self.template.point_counts, reference_labels=reference_labels)

    def to_dict(self, model_name: str, labels: List[str], reference_labels: List[str]) -> dict:
        """
        Convert the ActiveShapeModel to a JSON serializable dictionary.
        """
        return {
            'model_name': model_name,
            'mean': self.pca.mean_.tolist(),
            'components': self.pca.components_.tolist(),
            'explained_variance': self.pca.explained_variance_.tolist(),
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'template_points': self.template.points.tolist(),
            'template_reference': self.template.reference_points,
            'labels': labels,
            'reference_labels': reference_labels,
            'point_counts': self.template.point_counts
        }


def main():
    import argparse
    import sys

    from szkspyne import labelme_utils

    parser = argparse.ArgumentParser(
        description='Create an Active Shape Model from spine point data.')
    parser.add_argument('output', help='Output npz filename')
    parser.add_argument(
        '--input', type=argparse.FileType('r'), default=sys.stdin, help='Input ndjson filename'
    )
    parser.add_argument(
        '--model_name',
        help='Name of the model',
    )
    parser.add_argument(
        '--json',
        help="Save the model as a JSON file")
    parser.add_argument(
        '--labelme',
        help='Input is in labelme format',
        action='store_true',
    )

    args = parser.parse_args()

    if args.labelme:
        shape_dicts = [line.into_shape_dict_line() for line in labelme_utils.LabelMeLine.from_lines(args.input)]
    else:
        shape_dicts = labelme_utils.ShapeDictLine.from_lines(args.input)
    logger.info('Loaded %d shape dicts', len(shape_dicts))
    logger.info(f"Loading point data for {settings.ASM_LABEL_NAMES} and {settings.ANCHOR_LABEL_NAMES}")
    spine_data = [PresetPointData.from_shape_dict(shape_dict.content) for shape_dict in shape_dicts]
    logger.info('Point counts: %s', spine_data[0].point_counts)

    logger.info('Building Active Shape Model from %d spine data', len(spine_data))
    asm_builder = ActiveShapeModelBuilder.from_data(spine_data)
    # %%
    logger.info('Saving Active Shape Model to %s', args.output)
    if not args.model_name:
        from pathlib import Path
        args.model_name = Path(args.output).stem
    asm_builder.save(args.output, args.model_name, settings.ASM_LABEL_NAMES, settings.ANCHOR_LABEL_NAMES)
    logger.info('Active Shape Model saved successfully.')

    if args.json:
        import json
        json_output = asm_builder.to_dict(args.model_name, settings.ASM_LABEL_NAMES, settings.ANCHOR_LABEL_NAMES)
        with open(args.json, 'w') as f:
            json.dump(json_output, f, indent=4)
        logger.info('Active Shape Model saved as JSON to %s', args.json)

if __name__ == '__main__':
    main()