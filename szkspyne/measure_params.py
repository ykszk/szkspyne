from pathlib import Path
import json

from PIL import Image
import numpy as np
from logzero import logger
import sympy
from sympy.geometry import Point, Line, Polygon, Segment, Circle, intersection
from sympy import solve
from sympy.abc import x, y
import svgwrite
import yaml
from szkspyne import measure_utils as utils
from szkspyne.measure_utils import calc_centroid
import argparse

LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20) # perpendicular sign


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='argparse example.')  # TODO: change description
    parser.add_argument('input', help='Input labelme json filename', metavar='<input>', nargs='?', default='data/OSC00503_20210330_1.json')
    parser.add_argument('output', help='Output json filename', metavar='<output>', nargs='?', default='measurements.json')
    # parser.add_argument('-o',
    #                     '--option',
    #                     help='Optional argument. default: %(default)s',
    #                     metavar='<name>',
    #                     default='default')
    # parser.add_argument('-s',
    #                     '--switch',
    #                     help='Switch argument',
    #                     action='store_true')

    args, unknown = parser.parse_known_args()

    json_fn = Path(args.input)
    with open(json_fn, encoding='utf8') as f:
        lm_data = json.load(f)

    LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']
    measurements = {}
    points = utils.get_point_dict(lm_data)
    kps = utils.sort_kps(points)
    n_vertebrae = len(kps['BR'])
    sacrum_tltr, vert_corners, disc_corners = utils.get_vert_corners(kps)

    logger.info('Compute centroids')
    sacrum_center = sacrum_tltr.mean(axis=0)
    vert_centroids = np.array([calc_centroid(c) for c in vert_corners])
    disc_centroids = np.array([calc_centroid(c) for c in disc_corners])
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
    measurements['CobbPT'] = utils.calc_cobb(vert_sup_cnrs[1], vert_inf_cnrs[i_uiv])
    measurements['CobbMT'] = utils.calc_cobb(vert_sup_cnrs[i_uiv], vert_inf_cnrs[i_liv])
    measurements['CobbLT'] = utils.calc_cobb(vert_sup_cnrs[i_liv], vert_inf_cnrs[n_vertebrae - 2])

    # Apical Translation
    vert_apex = vert_centroids[i_apex]
    measurements['ApicalTranslation'] = vert_apex[0] - sacrum_center[0]

    # Coronal Balance
    c7_centroid = vert_centroids[0]
    measurements['CoronalBalance'] = c7_centroid[0] - sacrum_center[0]

    # T1 tilt
    HEIGHT, WIDTH = 1000, 1000
    c7_centroid = vert_centroids[0]
    c7pl = np.array((c7_centroid, (c7_centroid[0], HEIGHT)))
    t1_corners = vert_corners[1]
    t1_sup_vec = vert_sup_vecs[1]
    aux_vec = np.array([1, 0])
    t1_width = np.linalg.norm(t1_corners[1] - t1_corners[0])
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
    pelvic_cross = [
        float(e) for e in intersection(pelvic_ref_line, hline)[0].evalf()
    ]
    angle = hline.angle_between(pelvic_ref_line)
    if pelvic_kps[0, 1] > pelvic_kps[1, 1]:
        angle *= -1
    measurements['PelvicObliquity'] = np.rad2deg(float(angle.evalf()))

    # Leg Length Discrepancy
    femoral_kps = kps['FemoralHead']
    assert femoral_kps.shape == (2, 2)
    femoral_kps = np.array(sorted(femoral_kps, key=lambda e: e[0]))  # sort by x
    y_up = femoral_kps[:, 1].min()
    y_low = femoral_kps[:, 1].max()
    measurements['LegLengthDiscrepancy'] = femoral_kps[1, 1] - femoral_kps[0, 1]

    # Sacral Obliquity
    femoral_line = Line(femoral_kps[0], femoral_kps[1])
    sacrum_line = Line(sacrum_tltr[0], sacrum_tltr[1])
    x_pos1, x_pos2 = femoral_kps[:, 0]
    y_pos1 = float(solve(sacrum_line.equation(x, y).subs(x, x_pos1).evalf())[0])
    y_pos2 = float(solve(sacrum_line.equation(x, y).subs(x, x_pos2).evalf())[0])
    measurements['SacralObliquity'] = utils.calc_cobb(np.array([[x_pos1, y_pos1], [x_pos2, y_pos2]]), femoral_kps)

    with open(args.output, 'w') as f:
        json.dump(measurements, f, indent=False)