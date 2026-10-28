import numpy as np
from skimage.transform import SimilarityTransform

from szkspyne.asm.point_data import MovablePoints

REFS = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]])


def points_with_transform_params(scale, tx, ty, rotation):
    '''
    Points whose transform onto `REFS` has the given parameters
    '''
    tf = SimilarityTransform(scale=scale, rotation=rotation, translation=(tx, ty))
    refs = tf.inverse(REFS)
    return MovablePoints(points=refs, reference_points=list(refs), point_counts=[len(refs)])


def test_select_template_returns_sample_closest_to_mean_params():
    center = np.array([1.2, 10.0, -5.0, 0.1])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    offsets = [
        center - identity,
        np.array([0.05, 3.0, 1.0, 0.02]),
        np.array([-0.03, 1.0, 4.0, -0.05]),
        np.array([0.01, -2.0, 2.0, 0.03]),
        np.array([0.02, 1.0, -3.0, 0.01]),
    ]
    # Parameters symmetric around `center`, so the sample at `center` is the closest to the mean.
    # The first sample is the identity (the tentative template itself), the second is at `center`.
    params = [center - offsets[0], center, center + offsets[0]]
    for d in offsets[1:]:
        params += [center - d, center + d]
    mps = [points_with_transform_params(*p) for p in params]

    template = MovablePoints.select_template(mps)

    assert template is mps[1]
