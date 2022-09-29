#%%
from pathlib import Path
import json

import numpy as np
from logzero import logger
from sympy.geometry import Point, Line, Polygon, Segment, Circle

LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']
LINE_WIDTH = 2.0
PERP_BOX_SIZE = (20, 20)


json_fn = Path('data/OSC00525_20210803_15.json')
with open(json_fn, encoding='utf8') as f:
    lm_data = json.load(f)

measurements = {}
text_attrs = dict(stroke='black',
                  stroke_width='1px',
                  fill='white',
                  style="font-size:24px; font-family:sans-serif")
from szkspyne import measure_utils as utils
from szkspyne.measure_utils import calc_cobb, calc_centroid, get_circle_dict, perp

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
gid = 'SacralSlope'
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
l5_perp = perp(l5_sup[1] - l5_sup[0])
l5_width = np.linalg.norm(l5_sup[1] - l5_sup[0])
angle = Line(*l5_sup).angle_between(Line(femoral_center, l5_sup_center))
measurements['L5IncidenceAngle'] = 90 - np.rad2deg(float(angle.evalf()))

# Lumbosacral Angle
measurements['LumbosacralAngle'] = calc_cobb(vert_inf_cnrs[-1],
                sacrum_tltr)

with open('measurements_lateral.json', 'w') as f:
    json.dump(measurements, f, indent=False)