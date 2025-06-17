#%%
from typing import List, Optional
from szkspyne import labelme_utils
import numpy as np
from skimage.transform import SimilarityTransform, warp
from abc import ABC, abstractmethod


class MovablePoints(ABC):

    @abstractmethod
    def points(self) -> np.ndarray:
        """
        Returns the points of the movable object.
        The points are in the image coordinate system.
        The points are in the format (x, y).
        """
        pass

    @abstractmethod
    def reference_points(self) -> np.ndarray:
        """
        Returns the reference points of the movable object.
        The reference points are in the image coordinate system.
        The points are in the format (x, y).
        """
        pass

    @abstractmethod
    def incomplete_reference_points(self) -> List[Optional[float]]:
        """
        Returns the incomplete reference points of the movable object.
        The reference points are in the image coordinate system.
        The points are in the format (x, y).
        If a point is not available, it should return None.
        """
        pass


class TemplatePoints(MovablePoints):
    def calculate_transform(self, src_points: MovablePoints) -> SimilarityTransform:
        """
        Calculate the transform parameters to transform the src_points to the template points.
        The transform parameters are defined as (scale, rotation, translation).
        """
        src = src_points.reference_points()
        dst = self.reference_points()
        # calculate similarity transform parameters
        tf = SimilarityTransform()
        tf.estimate(src, dst)
        return tf
    
    def align(self, src_points: MovablePoints) -> tuple:
        """
        Align the src_points to the template points using the transform parameters.
        The transform parameters are defined as (scale, rotation, translation).
        """
        tf = self.calculate_transform(src_points)
        transformed_points = tf(src_points.points())
        return transformed_points, tf
    
    @staticmethod
    def select_template(mps: List['MovablePoints']) -> 'TemplatePoints':
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

class Spine():
    """
    Class to represent the spine in a DICOM image.
    The spine is represented by series of four corner points: TL, TR, BL, BR.
    The points are stored in a numpy array of shape (N, 4, 2).
    The points are in the order: TL, TR, BL, BR.
    The points are in the image coordinate system.
    The points are in the format (x, y).
    """
    vertebrae: np.ndarray

    def __init__(self, vertebrae: np.ndarray, shape_dict: labelme_utils.ShapeDict):
        self.vertebrae = vertebrae
        self.shape_dict = shape_dict

    @staticmethod
    def from_shape_dict(shape_dict: labelme_utils.ShapeDict):
        labels = ['TL', 'TR', 'BL', 'BR']
        corner_points = []
        for label in labels:
            if label not in shape_dict.shapes['point']:
                ValueError(f"Label {label} not found in shape_dict")
            points = np.array(shape_dict.shapes['point'][label])
            points = points[:, 0]
            # sort by y coordinate
            points = points[np.argsort(points[:, 1])]
            if label in ['TL', 'TR']:
                # remove sacrum points
                points = points[:-1]
            corner_points.append(points)
        return Spine(np.stack(corner_points, axis=1), shape_dict)
    
    def mean_plate_length(self):
        """
        Calculate the mean plate length of the spine.
        The mean plate length is defined as the average distance between the TL and TR points.
        """
        tl_tr = np.linalg.norm(self.vertebrae[:, 0] - self.vertebrae[:, 1], axis=1)
        bl_br = np.linalg.norm(self.vertebrae[:, 2] - self.vertebrae[:, 3], axis=1)
        return np.mean(np.concatenate([tl_tr, bl_br]))
    
    def crop_params(self):
        labels_lr_split = ['Clavicle','Shoulder','Pelvis','Iliac']
        left_points, right_points = [], []
        shape_dict = self.shape_dict
        width, height = shape_dict.imageWidth, shape_dict.imageHeight
        mean_plate_length = self.mean_plate_length()
        for label in labels_lr_split:
            if label not in shape_dict.shapes['point']:
                continue
            for point in shape_dict.shapes['point'][label]:
                point = point[0]
                if point[0] < width / 2:
                    left_points.append(point)
                else:
                    right_points.append(point)
        if left_points:
            left_points = np.array(left_points)
        else:
            left_points = np.zeros((0, 2))
        if right_points:
            right_points = np.array(right_points)
        else:
            right_points = np.zeros((0, 2))

        t1_top_middle = self.vertebrae[1, 0] + (self.vertebrae[1, 1] - self.vertebrae[1, 0]) / 2
        # t1_top_middle + 4*mean_plate_length
        est_right_point = t1_top_middle.copy()
        est_right_point[0] += 4 * mean_plate_length
        right_points = np.concatenate([right_points, [est_right_point]], axis=0)

        est_left_point = t1_top_middle.copy()
        est_left_point[0] -= 4 * mean_plate_length
        left_points = np.concatenate([left_points, [est_left_point]], axis=0)

        crop_x_right = np.clip(right_points.max(axis=0)[0], 0, width).astype(int)
        crop_x_left = np.clip(left_points.min(axis=0)[0], 0, width).astype(int)
        y_margin = mean_plate_length
        crop_y_top = np.clip(self.vertebrae.reshape((-1,2)).min(axis=0)[1] - y_margin, 0, height).astype(int)
        crop_y_bottom = np.clip(self.vertebrae.reshape((-1,2)).max(axis=0)[1] + y_margin, 0, height).astype(int)
        return crop_x_left, crop_x_right, crop_y_top, crop_y_bottom
    
    @staticmethod
    def crop(vertebrae: np.ndarray, image, crop_params):
        """
        Crop the image using the crop parameters.
        The crop parameters are defined as (crop_x_left, crop_x_right, crop_y_top, crop_y_bottom).
        """
        crop_x_left, crop_x_right, crop_y_top, crop_y_bottom = crop_params
        cropped_image = image[crop_y_top:crop_y_bottom, crop_x_left:crop_x_right]
        offsetted_spine = vertebrae.copy()
        offsetted_spine[:, :, 0] -= crop_x_left
        offsetted_spine[:, :, 1] -= crop_y_top
        return cropped_image, offsetted_spine
    
    @staticmethod
    def estimate_transform(src_spine: 'Spine', dst_spine: 'Spine') -> SimilarityTransform:
        """
        Calculate the transform parameters to transform the src_spine to the dst_spine.
        The transform parameters are defined as (scale, rotation, translation).
        """
        src_points = src_spine.vertebrae[[0,-1]].reshape(-1, 2)
        dst_points = dst_spine.vertebrae[[0,-1]].reshape(-1, 2)

        # calculate similarity transform parameters
        # that transform points1 to points2
        tf = SimilarityTransform()
        tf.estimate(src_points, dst_points)
        return tf
    
    class ReferenceSpine:
        """
        Reference spine for alignment.
        """
        def __init__(self, spine: 'Spine', image):
            self.spine = spine
            self.image = image
            self.crop_params = spine.crop_params()

        def align(self, src_spine: 'Spine', src_image) -> tuple:
            """
            Align the src_spine to the reference spine using the transform parameters.
            The transform parameters are defined as (scale, rotation, translation).
            """
            tf = Spine.estimate_transform(src_spine, self.spine)
            tfed_vertebrae = tf(src_spine.vertebrae.reshape(-1, 2)).reshape(src_spine.vertebrae.shape)
            tfed_shape_dict = src_spine.shape_dict.model_copy().transform(tf)

            # calculate output_shape
            corner_points = np.array([[0, 0], 
                                  [0, src_image.shape[0]],
                                  [src_image.shape[1], src_image.shape[0]],
                                  [src_image.shape[1], 0]])
            tfed_corner_points = tf(corner_points)
            tl_br = np.array([tfed_corner_points.min(axis=0), tfed_corner_points.max(axis=0)])
            output_shape = (int(tl_br[1, 1] - min(0, tl_br[0, 1])), int(tl_br[1, 0] - min(0, tl_br[0, 0])))
            warped_image = warp(src_image, tf.inverse, output_shape=output_shape, order=1)
            # offset the transformed spine
            cropped_warped_image, offsetted_spine_transformed = Spine.crop(tfed_vertebrae, warped_image, self.crop_params)
            tl = np.array([self.crop_params[0], self.crop_params[2]])
            crop_tf = lambda points: points - tl
            tfed_shape_dict = tfed_shape_dict.transform(crop_tf)
            return offsetted_spine_transformed, cropped_warped_image, tfed_shape_dict
        
        def crop_self(self) -> tuple:
            """
            Crop the reference spine from the image using the crop parameters.
            The crop parameters are defined as (crop_x_left, crop_x_right, crop_y_top, crop_y_bottom).
            """
            cropped_image, offsetted_spine = Spine.crop(self.spine.vertebrae, self.image, self.crop_params)
            return offsetted_spine, cropped_image
    
    @staticmethod
    def align_spines(src_spine: 'Spine', src_image, dst_spine: 'Spine', dst_image) -> tuple:
        """
        Align the src_spine to the dst_spine using the transform parameters.
        The transform parameters are defined as (scale, rotation, translation).
        """
        ref_spine = Spine.ReferenceSpine(dst_spine, dst_image)

        aligned_src_spine, aliend_src_image = ref_spine.align(src_spine, src_image)
        aligned_dst_spine, aligned_dst_image = ref_spine.crop_self()
        return aligned_src_spine, aliend_src_image, aligned_dst_spine, aligned_dst_image
