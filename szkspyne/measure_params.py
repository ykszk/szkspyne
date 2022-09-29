from pathlib import Path
import json

import numpy as np
from logzero import logger
from sympy.geometry import Line, intersection
from sympy import solve
from sympy.abc import x, y
from szkspyne import measure_utils as utils
from szkspyne.measure_utils import calc_cobb, calc_centroid, get_circle_dict, perp
import argparse
LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']
LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20) # perpendicular sign
HEIGHT, WIDTH = 1000, 1000


def measure_ap(lm_data: dict) -> dict:
    measurements = {}
    points = utils.get_point_dict(lm_data)
    kps = utils.sort_kps(points)
    n_vertebrae = len(kps['BR'])
    sacrum_tltr, vert_corners, disc_corners = utils.get_vert_corners(kps)

    logger.info('Compute centroids')
    sacrum_center = sacrum_tltr.mean(axis=0)
    vert_centroids = np.array([calc_centroid(c) for c in vert_corners])
    logger.info('Compute vecs')
    vert_sup_cnrs = vert_corners[:, :2]
    vert_sup_vecs = vert_sup_cnrs[:, 1] - vert_sup_cnrs[:, 0]
    vert_sup_vecs /= np.linalg.norm(vert_sup_vecs, axis=1, keepdims=True)
    vert_inf_cnrs = vert_corners[:, 2:]
    vert_inf_vecs = vert_inf_cnrs[:, 1] - vert_inf_cnrs[:, 0]
    vert_inf_vecs /= np.linalg.norm(vert_inf_vecs, axis=1, keepdims=True)

    logger.info('Coronal translation')
    coronal_translation = vert_centroids[:, 0] - sacrum_center[0]
    i_apex = np.abs(coronal_translation).argmax()
    i_uiv = np.abs(vert_sup_vecs[:i_apex, 1]).argmax()
    i_liv = np.abs(vert_inf_vecs[(i_apex + 1):, 1]).argmax() + i_apex + 1

    # cobb angles
    logger.info('Cobb angles')
    measurements['CobbPT'] = calc_cobb(vert_sup_cnrs[1], vert_inf_cnrs[i_uiv])
    measurements['CobbMT'] = calc_cobb(vert_sup_cnrs[i_uiv], vert_inf_cnrs[i_liv])
    measurements['CobbLT'] = calc_cobb(vert_sup_cnrs[i_liv], vert_inf_cnrs[n_vertebrae - 2])

    # Apical Translation
    vert_apex = vert_centroids[i_apex]
    measurements['ApicalTranslation'] = vert_apex[0] - sacrum_center[0]

    # Coronal Balance
    c7_centroid = vert_centroids[0]
    measurements['CoronalBalance'] = c7_centroid[0] - sacrum_center[0]

    # T1 tilt
    c7_centroid = vert_centroids[0]
    c7pl = np.array((c7_centroid, (c7_centroid[0], HEIGHT)))
    line_t1 = Line(vert_corners[1, 0], vert_corners[1, 1])
    aux_t1_tilt = np.array([
        float(e) for e in intersection(Line(c7pl[0], c7pl[1]), line_t1)[0].evalf()
    ])
    angle = line_t1.angle_between(
        Line([0, aux_t1_tilt[1]], [WIDTH, aux_t1_tilt[1]]))
    measurements['T1Tilt'] = np.rad2deg(float(angle.evalf()))

    # Clavicle Angle
    assert kps['Clavicle'].shape == (2, 2)
    clavicle_kps = kps['Clavicle']
    line_clavicle = Line(clavicle_kps[0], clavicle_kps[1])
    chrl = Line([0, clavicle_kps[:, 1].min()],
                [WIDTH, clavicle_kps[:, 1].min()])

    angle = chrl.angle_between(line_clavicle)
    measurements['ClavicleAngle'] = np.rad2deg(float(angle.evalf()))

    # Shoulder Height
    shoulder_kps = kps['Shoulder']
    shoulder_kps = np.array(sorted(shoulder_kps, key=lambda e: e[0]))  # sort by x
    assert shoulder_kps.shape == (2, 2)
    measurements['ShoulderHeight'] = shoulder_kps[0, 1] - shoulder_kps[1, 1]

    # Pelvic Obliquity
    pelvic_kps = kps['Pelvis']
    assert pelvic_kps.shape == (2, 2)
    pelvic_kps = np.array(sorted(pelvic_kps, key=lambda e: e[0]))  # sort by x
    pelvic_ref_line = Line(pelvic_kps[0], pelvic_kps[1])
    y_up = pelvic_kps[:, 1].min()
    hline = Line((0, y_up), (WIDTH, y_up))
    angle = hline.angle_between(pelvic_ref_line)
    if pelvic_kps[0, 1] > pelvic_kps[1, 1]:
        angle *= -1
    measurements['PelvicObliquity'] = np.rad2deg(float(angle.evalf()))

    # Leg Length Discrepancy
    femoral_kps = kps['FemoralHead']
    assert femoral_kps.shape == (2, 2)
    femoral_kps = np.array(sorted(femoral_kps, key=lambda e: e[0]))  # sort by x
    y_up = femoral_kps[:, 1].min()
    measurements['LegLengthDiscrepancy'] = femoral_kps[1, 1] - femoral_kps[0, 1]

    # Sacral Obliquity
    sacrum_line = Line(sacrum_tltr[0], sacrum_tltr[1])
    x_pos1, x_pos2 = femoral_kps[:, 0]
    y_pos1 = float(solve(sacrum_line.equation(x, y).subs(x, x_pos1).evalf())[0])
    y_pos2 = float(solve(sacrum_line.equation(x, y).subs(x, x_pos2).evalf())[0])
    measurements['SacralObliquity'] = calc_cobb(np.array([[x_pos1, y_pos1], [x_pos2, y_pos2]]), femoral_kps)

    return measurements

def measure_lateral(lm_data: dict) -> dict:
    measurements = {}
    points = utils.get_point_dict(lm_data)
    kps = utils.sort_kps(points)
    sacrum_tltr, vert_corners, disc_corners = utils.get_vert_corners(kps)

    logger.info('Compute centroids')
    sacrum_center = sacrum_tltr.mean(axis=0)
    logger.info('Compute vecs')
    vert_sup_cnrs = vert_corners[:, :2]
    vert_sup_vecs = vert_sup_cnrs[:, 1] - vert_sup_cnrs[:, 0]
    vert_sup_vecs /= np.linalg.norm(vert_sup_vecs, axis=1, keepdims=True)
    vert_inf_cnrs = vert_corners[:, 2:]
    vert_inf_vecs = vert_inf_cnrs[:, 1] - vert_inf_cnrs[:, 0]
    vert_inf_vecs /= np.linalg.norm(vert_inf_vecs, axis=1, keepdims=True)


    # Thoracic kyphosis
    sup = vert_sup_cnrs[2]
    inf = vert_inf_cnrs[12]
    measurements['Thoracickyphosis'] = calc_cobb(sup, inf)

    # Mid/Lower thoracic kyphosis
    sup = vert_sup_cnrs[5]
    inf = vert_inf_cnrs[12]
    measurements['Mid/Lowerthoracickyphosis'] = calc_cobb(sup, inf)

    # Proximal thoracic kyphosis
    sup = vert_sup_cnrs[2]
    inf = vert_inf_cnrs[5]
    measurements['Proximalthoracickyphosis'] = calc_cobb(sup, inf)

    # Pelvic Incidence
    circles = get_circle_dict(lm_data)
    femoral_heads = np.array(circles['FemoralHead'])
    femoral_center = femoral_heads[:, 0].mean(axis=0)
    angle = Line(*sacrum_tltr).angle_between(Line(femoral_center, sacrum_center))
    measurements['PelvicIncidence'] = 90 - np.rad2deg(float(angle.evalf()))

    # Sacral Slope
    sacrum_width = np.linalg.norm(sacrum_tltr[1] - sacrum_tltr[0])
    sacrum_hl = sacrum_tltr[0] + sacrum_width * np.array(
        (np.sign(sacrum_tltr[1, 0] - sacrum_tltr[0, 0]), 0))
    angle = Line(*sacrum_tltr).angle_between(Line(sacrum_tltr[0], sacrum_hl))
    measurements['SacralSlope'] = 90 - np.rad2deg(float(angle.evalf()))

    # Pelvic Radius Angle
    angle = Line(*sacrum_tltr).angle_between(Line(femoral_center, sacrum_tltr[1]))
    measurements['PelvicRadiusAngle'] = (np.rad2deg(float(angle.evalf())))

    # L5 Incidence Angle
    l5_sup = vert_sup_cnrs[-1]
    l5_sup_center = l5_sup.mean(axis=0)
    angle = Line(*l5_sup).angle_between(Line(femoral_center, l5_sup_center))
    measurements['L5IncidenceAngle'] = 90 - np.rad2deg(float(angle.evalf()))

    # Lumbosacral Angle
    measurements['LumbosacralAngle'] = calc_cobb(vert_inf_cnrs[-1],
                    sacrum_tltr)
    return measurements

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Measure scoliosis parameters from the given labelme annotation file.')
    parser.add_argument('input', help='Input labelme json filename', metavar='<input>')
    parser.add_argument('output', help='Output json filename', metavar='<output>')
    parser.add_argument('--image_type',
                        help='Image type. default: %(default)s',
                        choices=['ap', 'lateral'],
                        metavar='<name>',
                        default='ap')

    args, unknown = parser.parse_known_args()

    json_fn = Path(args.input)
    with open(json_fn, encoding='utf8') as f:
        lm_data = json.load(f)

    if args.image_type == 'ap':
        measurements = measure_ap(lm_data)
    else:
        measurements = measure_lateral(lm_data)
    with open(args.output, 'w') as f:
        json.dump(measurements, f, indent=False)