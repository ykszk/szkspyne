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
from measure_utils import calc_centroid, draw_cobb

# json_fn = Path('../../data/annotation/reference/OSC00524_20210730_1.json')
# json_fn = Path('data/OSC00524_20210730_1.json')
json_fn = Path('data/OSC00503_20210330_1.json')
with open(json_fn, encoding='utf8') as f:
    lm_data = json.load(f)

img_fn = json_fn.parent / lm_data['imagePath']
img = Image.open(img_fn)
with open('data/config.yaml') as f:
    color_data = yaml.safe_load(f)
label_colors = color_data['label_colors']
#%%
LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']

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

#%% coronal translation
logger.info('Coronal translation')
coronal_translation = vert_centroids[:, 0] - sacrum_center[0]
i_apex = np.abs(coronal_translation).argmax()
i_uiv = np.abs(vert_sup_vecs[:i_apex, 1]).argmax()
i_liv = np.abs(vert_inf_vecs[(i_apex + 1):, 1]).argmax() + i_apex + 1

text_attrs = dict(stroke='black',
                  stroke_width='1px',
                  fill='white',
                  style="font-size:24px; font-family:sans-serif")
LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20) # perpendicular sign




w, h = img.size
pixel_size = 1
logger.info('drawing')
dwg = svgwrite.Drawing('output.svg',
                       size=(w * pixel_size, h * pixel_size),
                       viewBox=('0 0 {} {}'.format(w, h)))
logger.info('image')
dwg.add(dwg.image('data:image/jpg;base64,' + utils.img2b64(img), size=img.size))
drawings = []
# points
if False:
    for label, pts in points.items():
        grp = dwg.add(dwg.g(id=label, stroke_width=2.0, fill='none'))
        color = '#' + ''.join(['{:02x}'.format(c) for c in label_colors[label]])
        for pt in pts:
            grp.add(dwg.circle(center=pt, r=POINT_RADIUS, stroke=color, fill=color))
    dwg.save()
    dwg
#%%
# centroids
if False:
    drawings.append(('Centroids', 'Centroids'))
    color = 'green'
    grp = dwg.add(
        dwg.g(id='Centroids', stroke=color, stroke_width=2.0, fill='none'))
    for c in vert_centroids:
        grp.add(dwg.circle(center=c.tolist(), r=POINT_RADIUS, stroke=color, fill=color))

# cobb angles
logger.info('Cobb angles')
drawings += [('PT', 'PT Cobb (p.49)'), ('MT', 'MT Cobb (p.49)'),
             ('LT', 'LT Cobb (p.49)')]
draw_cobb(vert_sup_cnrs[1], vert_inf_cnrs[i_uiv], dwg, 'PT', 'red', text_attrs=text_attrs)
draw_cobb(vert_sup_cnrs[i_uiv], vert_inf_cnrs[i_liv], dwg, 'MT', 'green', text_attrs=text_attrs)
draw_cobb(vert_sup_cnrs[i_liv], vert_inf_cnrs[n_vertebrae - 2], dwg, 'LT',
          'blue', text_attrs=text_attrs)
# apical translation
drawings.append(('APT', 'Apical Translation (p.53)'))
logger.info(drawings[-1][1])
vert_apex = vert_centroids[i_apex]
color = 'green'
grp = dwg.add(dwg.g(id='APT', stroke=color, stroke_width=2.0, fill='none'))
grp.add(
    dwg.path((['M'] + sacrum_center.tolist() + [sacrum_center[0]] +
              [vert_corners[:, :, 1].min()])))
grp.add(
    dwg.path((['M'] + vert_apex.tolist() + [sacrum_center[0]] +
              [vert_centroids[i_apex, 1]])))
grp.add(dwg.circle(center=vert_apex.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
grp.add(
    dwg.text('{:.1f} pixels'.format(vert_apex[0] - sacrum_center[0]),
             insert=vert_apex,
             **text_attrs))
grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
for pos in np.concatenate((sacrum_tltr, sacrum_center[np.newaxis]), axis=0):
    grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))

# coronal balance
drawings.append(('CoronalBalance', 'Coronal Balance (p.54)'))
logger.info(drawings[-1][1])
color = 'blue'
grp = dwg.add(
    dwg.g(id='CoronalBalance', stroke=color, stroke_width=2.0, fill='none'))
c7_centroid = vert_centroids[0]


def lerp(a, b, t):
    assert 0 <= t <= 1
    return t * a + (1 - t) * b


grp.add(
    dwg.path((['M'] + sacrum_center.tolist() + [sacrum_center[0]] +
              [lerp(sacrum_center[1], c7_centroid[1], .4)])))
grp.add(
    dwg.path((['M'] + c7_centroid.tolist() + [c7_centroid[0]] +
              [lerp(sacrum_center[1], c7_centroid[1], .6)])))
grp.add(
    dwg.circle(center=sacrum_center.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
grp.add(dwg.circle(center=c7_centroid.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
y_cb = lerp(sacrum_center[1], c7_centroid[1], .5)
grp.add(dwg.path((['M'] + [c7_centroid[0], y_cb] + [sacrum_center[0], y_cb])))
grp.add(
    dwg.text('{:.1f} pixels'.format(c7_centroid[0] - sacrum_center[0]),
             insert=[max(c7_centroid[0], sacrum_center[0]), y_cb],
             **text_attrs))
# T1 tilt
drawings.append(('T1Tilt', 'T1 Tilt Angle (p.55)'))
logger.info(drawings[-1][1])
color = 'blue'
grp = dwg.add(dwg.g(id='T1Tilt', stroke=color, stroke_width=2.0, fill='none'))
c7_centroid = vert_centroids[0]
c7pl = np.array((c7_centroid, (c7_centroid[0], img.size[1])))
t1_corners = vert_corners[1]
t1_sup_vec = vert_sup_vecs[1]
aux_vec = np.array([1, 0])
t1_width = np.linalg.norm(t1_corners[1] - t1_corners[0])
line_t1 = Line(vert_corners[1, 0], vert_corners[1, 1])
aux_t1_tilt = np.array([
    float(e) for e in intersection(Line(c7pl[0], c7pl[1]), line_t1)[0].evalf()
])
grp.add(
    dwg.path((['M'] + (aux_t1_tilt - 2 * t1_width * t1_sup_vec).tolist() +
              (aux_t1_tilt + 5 * t1_width * t1_sup_vec).tolist())))
grp.add(
    dwg.path((['M'] + (aux_t1_tilt - 2 * t1_width * aux_vec).tolist() +
              (aux_t1_tilt + 5 * t1_width * aux_vec).tolist())))
angle = line_t1.angle_between(
    Line([0, aux_t1_tilt[1]], [img.size[0], aux_t1_tilt[1]]))
txt_angle = '{:.2f}°'.format(np.rad2deg(float(angle.evalf())))
grp.add(dwg.text(txt_angle, insert=c7_centroid, **text_attrs))
# Clavicle Angle
drawings.append(('ClavicleAngle', 'Clavicle Angle (p.56)'))
logger.info(drawings[-1][1])
color = 'green'
grp = dwg.add(
    dwg.g(id='ClavicleAngle', stroke=color, stroke_width=2.0, fill='none'))
assert kps['Clavicle'].shape == (2, 2)
clavicle_kps = kps['Clavicle']
line_clavicle = Line(clavicle_kps[0], clavicle_kps[1])
chrl = Line([0, clavicle_kps[:, 1].min()],
            [img.size[0], clavicle_kps[:, 1].min()])
t = sympy.Symbol('t', real=True)
grp.add(
    dwg.path((['M'] + [
        0, float(solve(line_clavicle.equation(x, y).subs(x, 0).evalf())[0])
    ] + [
        img.size[0],
        float(
            solve(line_clavicle.equation(x, y).subs(x,
                                                    img.size[0]).evalf())[0])
    ])))

clavicle_intersection = [
    float(e) for e in intersection(chrl, line_clavicle)[0].evalf()
]
grp.add(
    dwg.path((['M'] + [0, clavicle_kps[:, 1].min()] +
              [img.size[0], clavicle_kps[:, 1].min()])))
# grp.add(dwg.circle(center=clavicle_intersection, r=POINT_RADIUS, stroke=color,
#                    fill=color))

angle = chrl.angle_between(line_clavicle)
txt_angle = '{:.1f}°'.format(np.rad2deg(float(angle.evalf())))
grp.add(dwg.text(txt_angle, insert=clavicle_intersection, **text_attrs))
for pos in clavicle_kps:
    grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
# Shoulder Height
drawings.append(('ShoulderHeight', 'Shoulder Height (p.57)'))
logger.info(drawings[-1][1])
color = 'magenta'
grp = dwg.add(
    dwg.g(id='ShoulderHeight', stroke=color, stroke_width=2.0, fill='none'))
shoulder_kps = kps['Shoulder']
shoulder_kps = np.array(sorted(shoulder_kps, key=lambda e: e[0]))  # sort by x
assert shoulder_kps.shape == (2, 2)
grp.add(
    dwg.path(
        (['M'] + [0, shoulder_kps[0, 1]] + [img.size[0], shoulder_kps[0, 1]])))
grp.add(
    dwg.path(
        (['M'] + [0, shoulder_kps[1, 1]] + [img.size[0], shoulder_kps[1, 1]])))
height_text = '{:.1f} pixels'.format(shoulder_kps[0, 1] - shoulder_kps[1, 1])
grp.add(dwg.text(height_text, insert=shoulder_kps[0], **text_attrs))
for pos in shoulder_kps:
    grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))

# Coronal Position of LIV
if False:
    drawings.append(('CoronalLIVPos', 'Coronal Position of LIV (p.60)'))
    logger.info(drawings[-1][1])
    color = 'magenta'
    grp = dwg.add(
        dwg.g(id='CoronalLIVPos', stroke=color, stroke_width=2.0, fill='none'))
    grp.add(
        dwg.path((['M'] + sacrum_center.tolist() +
                  [sacrum_center[0], vert_centroids[i_liv][1]])))
    grp.add(
        dwg.path((['M'] + vert_centroids[i_liv].tolist() +
                  [sacrum_center[0], vert_centroids[i_liv][1]])))
    grp.add(
        dwg.circle(center=vert_centroids[i_liv].tolist(),
                   r=POINT_RADIUS,
                   stroke=color,
                   fill=color))
    grp.add(
        dwg.text('{:.1f} pixels'.format(vert_centroids[i_liv, 0] -
                                        sacrum_center[0]),
                 insert=vert_centroids[i_liv],
                 **text_attrs))
    grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
    for pos in np.concatenate((sacrum_tltr, sacrum_center[np.newaxis]),
                              axis=0):
        grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))

# Coronal Angulation of Disc Below LIV
if False:
    drawings.append(
        ('CoronalDiscAngulation', 'Coronal Disc Angulation below LIV (p.61)'))
    logger.info(drawings[-1][1])
    draw_cobb(vert_inf_cnrs[i_liv], vert_sup_cnrs[i_liv + 1], dwg,
              'CoronalDiscAngulation', 'cyan', text_attrs=text_attrs)

# Pelvic Obliquity
drawings.append(('PelvicObliquity', 'Pelvic Obliquity (p.93)'))
logger.info(drawings[-1][1])
color = 'cyan'
grp = dwg.add(
    dwg.g(id='PelvicObliquity',
          stroke=color,
          stroke_width=LINE_WIDTH,
          fill='none'))
pelvic_kps = kps['Pelvis']
assert pelvic_kps.shape == (2, 2)
pelvic_kps = np.array(sorted(pelvic_kps, key=lambda e: e[0]))  # sort by x
pelvic_ref_line = Line(pelvic_kps[0], pelvic_kps[1])
grp.add(
    dwg.path((['M'] + [
        0,
        float(solve(pelvic_ref_line.equation(x, y).subs(x, 0).evalf())[0])
    ] + [
        img.size[0],
        float(
            solve(pelvic_ref_line.equation(x, y).subs(x,
                                                      img.size[0]).evalf())[0])
    ])))
y_up = pelvic_kps[:, 1].min()
grp.add(dwg.path((['M'] + [0, y_up] + [img.size[0], y_up])))
hline = Line((0, y_up), (img.size[0], y_up))
pelvic_cross = [
    float(e) for e in intersection(pelvic_ref_line, hline)[0].evalf()
]
angle = hline.angle_between(pelvic_ref_line)
if pelvic_kps[0, 1] > pelvic_kps[1, 1]:
    angle *= -1
txt_angle = '{:.1f}°'.format(np.rad2deg(float(angle.evalf())))
grp.add(dwg.text(txt_angle, insert=pelvic_cross, **text_attrs))
for pos in pelvic_kps:
    grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
grp.add(dwg.rect(insert=(-LINE_WIDTH, y_up), size=PERP_BOX_SIZE))

# Leg Length Discrepancy
drawings.append(('LegLength', 'Leg Length Discrepancy (p.93)'))
logger.info(drawings[-1][1])
color = 'lime'
grp = dwg.add(
    dwg.g(id='LegLength', stroke=color, stroke_width=2.0, fill='none'))
femoral_kps = kps['FemoralHead']
assert femoral_kps.shape == (2, 2)
femoral_kps = np.array(sorted(femoral_kps, key=lambda e: e[0]))  # sort by x
y_up = femoral_kps[:, 1].min()
grp.add(dwg.path((['M'] + [0, y_up] + [img.size[0], y_up])))
y_low = femoral_kps[:, 1].max()
grp.add(dwg.path((['M'] + [0, y_low] + [img.size[0], y_low])))
grp.add(
    dwg.path((['M'] + femoral_kps[1].tolist() +
              [femoral_kps[1, 0], femoral_kps[0, 1]])))
height_text = '{:.1f} pixels'.format(femoral_kps[1, 1] - femoral_kps[0, 1])
grp.add(dwg.text(height_text, insert=femoral_kps[1], **text_attrs))
for pos in femoral_kps:
    grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))

# Sacral Obliquity
drawings.append(('SacralObliquity', 'Sacral Obliquity (p.94)'))
logger.info(drawings[-1][1])
color = 'hotpink'
femoral_line = Line(femoral_kps[0], femoral_kps[1])
sacrum_line = Line(sacrum_tltr[0], sacrum_tltr[1])
x_pos1, x_pos2 = femoral_kps[:, 0]
y_pos1 = float(solve(sacrum_line.equation(x, y).subs(x, x_pos1).evalf())[0])
y_pos2 = float(solve(sacrum_line.equation(x, y).subs(x, x_pos2).evalf())[0])
grp = draw_cobb(np.array([[x_pos1, y_pos1], [x_pos2, y_pos2]]), femoral_kps,
                dwg, 'SacralObliquity', color, 1, text_attrs)
for pos in np.concatenate((femoral_kps, sacrum_tltr), axis=0):
    grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))

# save
dwg.save()
dwg
#%%
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
template = env.get_template('template.html')
t_cb = env.get_template('t_checkbox.html')
cbs = []
for name, label in drawings:
    cbs.append(t_cb.render(name=name, label=label))
with open('ap_measure.html', 'w', encoding='utf8') as f:
    f.write(template.render(svg=dwg.tostring(), checkboxes='\n'.join(cbs)))
# %%
