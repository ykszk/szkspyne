import sys
import argparse
import random

import albumentations as A
import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# https://albumentations.ai/docs/examples/example_keypoints/
def vis_keypoints(image, keypoints, color=(0, 255, 0), diameter=15):
    image = image.copy()

    for (y, x) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)

    return image


class CoarseDropout(A.CoarseDropout):
    '''
    Apply identity transform to keypoints
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

img_tfs = [
    # CoarseDropout(
    #     max_holes=50, max_height=64, max_width=64, fill_value=.5, p=.2),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=.5),
    A.Affine(scale={
        'x': (.8, 1.2),
        'y': (.8, 1.2)
    },
             rotate=[-30, 30],
             shear=[-20, 20],
             p=.8),
    A.GaussNoise(var_limit=.02, p=.5)
]
album_transform = A.Compose(img_tfs,
                            keypoint_params=A.KeypointParams(
                                format='yx', remove_invisible=False))

album_transform_wo_kps = A.Compose(img_tfs)

def percentile_normalize(x, percentile, clip=False):
    minmax = np.percentile(
        x.ravel()[::10],
        [percentile, 100 - percentile])  # ::10 to speedup computation
    x = (x - minmax[0]) / (minmax[1] - minmax[0])
    if clip:
        np.clip(x, 0, 1, out=x)
    return x


def base_transform(x, kps=None):
    if x.dtype == np.uint8:
        x = x / 255
    else:
        x = percentile_normalize(x, .001)
    if x.ndim == 2:
        x = x[np.newaxis]
    x = x.astype(np.float32)
    if kps is None:
        return x
    else:
        return x, kps


def invert_base(x):
    x = np.clip(x, 0, 1)
    x = np.round(x * 255, out=x)
    return x.astype(np.uint8)


def random_screw(img_shape, keypoints):
    bg = np.zeros(img_shape, dtype=np.uint8)
    kps = np.array(keypoints).reshape((-1, 2, 2))
    width = 50
    for kp_pair in kps:
        d = kp_pair[0] - kp_pair[1]
        p0 = kp_pair[1].copy()
        p0[1] += width
        p1 = kp_pair[1] + width
        p2 = p0 + d
        pts = np.array([p0, p1, p2], np.int32)
        pts = np.flip(pts, axis=1)
        bg = cv2.fillPoly(bg, [pts], color=1)
    return bg


def mask_keypoints(img, keypoints, min_margin=40):
    # margin = np.abs(np.random.randn()) * min_margin + min_margin
    masked = np.copy(img)
    kp_pairs = np.array(keypoints).reshape(-1, 2, 2)
    for kps in kp_pairs:
        margin = np.linalg.norm(kps[0]-kps[1]) * np.random.rand()
        top, left = kps.min(axis=0) - margin
        bottom, right = kps.max(axis=0) + margin
        top, left = np.maximum((top, left), (0, 0))
        bottom, right = np.minimum((bottom, right), img.shape)
        top, left, bottom, right = [int(v) for v in [top, left, bottom, right]]
        slices = (slice(top, bottom), slice(left, right))
        alpha = np.random.rand()
        masked[slices] = alpha * masked[slices] + (1-alpha)*np.random.rand()
    return masked

def sholder_occlusion(x, keypoints):
    kp_pairs = np.array(keypoints).reshape(-1, 2, 2)
    y = int(kp_pairs[-1, :, 0].min() - .5 * np.linalg.norm(kp_pairs[-1, 0] - kp_pairs[-1, 1]))
    mm = x.min(), x.max()
    x = x.astype(np.float32)
    x[y:] += .5 * x.max()
    x = np.clip(x, mm[0], mm[1])
    return x, keypoints

import itertools
def random_warp(x, keypoints):
    '''
    keypoints: (y, x) key points
    d_max: maximum displacement
    '''
    # return x, keypoints
    pts = np.flip(np.array(keypoints), axis=1)  # to (x, y) points
    top_pair, bottom_pair = pts[:2], pts[2:4]
    step = .2
    ratio = np.arange(-step, 1+2*step, step)[:, np.newaxis, np.newaxis]
    # print(ratio)
    ctr_points = (top_pair * ratio + bottom_pair * (1-ratio)).reshape((-1,2))
    anchors = []
    for ps in itertools.product([0, x.shape[1]/2, x.shape[1]], [0, x.shape[0]/2, x.shape[0]]):
        if ps[0]==x.shape[1]/2 and ps[1]==x.shape[0]/2:
            pass
        else:
            anchors.append(ps)
    anchors = np.array(anchors)
    sshape = ctr_points.copy()
    tshape = ctr_points.copy()
    d_max = .8 * np.linalg.norm(top_pair[0] - top_pair[1])
    tshape += np.random.rand(*tshape.shape) * d_max - d_max / 2
    new_keypoints = tshape[[2,3,-4,-3],:].copy()
    sshape = np.concatenate([anchors, sshape], axis=0)
    tshape = np.concatenate([anchors, tshape], axis=0)
    matches = [cv2.DMatch(i,i,0) for i in range(len(sshape))]
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(tshape[np.newaxis],sshape[np.newaxis],matches)
    warped = tps.warpImage(x)

    new_keypoints = np.flip(new_keypoints, axis=1) # back to (y, x) points
    return warped, new_keypoints

def random_transform(x, keypoints):
    clipLimit = random.choice([10, 100])
    gridSize = random.choice([16, 32])
    x = apply_clahe(x, clipLimit, (gridSize, gridSize))
    # if np.random.rand() < .1:
    #     screw = random_screw(x.shape, keypoints)
    #     x[screw] = x.max()
    # if np.random.rand() < .2:
    #     x, keypoints = sholder_occlusion(x, keypoints)
    # if np.random.rand() < .2:
    #     x = mask_keypoints(x, keypoints)
    # if np.random.rand() < .2:
    #     x, keypoints = random_warp(x, keypoints)
    x, keypoints = base_transform(x, keypoints)
    x = x.transpose((1, 2, 0))  # to channel last
    tfed = album_transform(image=x, keypoints=keypoints)
    x, keypoints = tfed['image'], tfed['keypoints']
    x = x.transpose((2, 0, 1))  # to channel first
    return x.astype(np.float32), keypoints

def random_transform_wo_kps(x):
    x = base_transform(x)
    x = x.transpose((1, 2, 0))  # to channel last
    tfed = album_transform_wo_kps(image=x)
    x = tfed['image']
    x = x.transpose((2, 0, 1))  # to channel first
    return x.astype(np.float32)

# def random_transform_wo_kps(x):
#     x = base_transform(x)
#     x = x.transpose((1, 2, 0))  # to channel last
#     tfed = album_transform_wo_kps(image=x)
#     x = tfed['image']
#     x = x.transpose((2, 0, 1))  # to channel first
#     return x.astype(np.float32)

def apply_clahe(img, clipLimit=100, tileGridSize=(32, 32)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    # clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(32, 32))
    cl = clahe.apply(img)
    cl = cl - cl.min()
    cl = cl / cl.max()
    return np.round(np.clip(cl,0,1) * 255).astype(np.uint8)

# from PIL import Image
# from matplotlib.pyplot import rcParams
# rcParams['figure.figsize'] = 10,10
# import matplotlib.pyplot as plt
# img = Image.open(r'\\Poseidon\Users\suzuki\exp\spine\data\annotation\OUCS0001\20201014\0.png')
# dim = img.size
# img.thumbnail((2000,768))
# # img = Image.open('../data/png/OUCS0001/20201014/0.png')
# #img = img.convert("RGB")
# img = np.array(img)
# #img = img[np.newaxis]
# # orig_kp = [[600,700],[650,820],[1220,500],[1210,700]]
# orig_kp = np.flip(np.array([[683, 633],[824,654],[613,1381],[757,1348]]), axis=-1)
# orig_kp = orig_kp * (768 / dim[1])
# # vis = vis_keypoints(img, kp)
# # plt.imshow(vis)
# # plt.show()

# #x, kp = random_transform(img.transpose((2, 0, 1)), kp)
# x, kp = random_transform(img, orig_kp)
# x = np.squeeze(x)
# #x = x.transpose((1, 2, 0))  # to channel last

# x = np.clip(np.round(x*255), 0, 255).astype(np.uint8)
# # vis = vis_keypoints(x, kp)
# # plt.imshow(vis)
# kp = np.array(kp)
# plt.imshow(x, cmap='gray')
# plt.scatter(kp[:, 1], kp[:, 0])
# plt.show()

# Image.fromarray(vis).save('tmp.png')

# def main():
#    parser = argparse.ArgumentParser(description='Try random transform.')
#    parser.add_argument('input', help='Input image filename:',metavar='<input>')
#    parser.add_argument('points', help='Input pointset file. Simple test file or labelme json file.',metavar='<output>')
#    parser.add_argument('output', help='Output image filename',metavar='<output>')

#    args = parser.parse_args()

#    print(args.switch)

# if __name__ == '__main__':
#    sys.exit(main())
