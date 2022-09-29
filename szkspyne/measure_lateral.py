#%%
from pathlib import Path
import json

from PIL import Image
import numpy as np
from logzero import logger
import sympy
from sympy.geometry import Point, Line, Polygon, Segment, Circle
from sympy.abc import x, y
import svgwrite

json_fn = Path('data/OSC00525_20210803_15.json')
with open(json_fn, encoding='utf8') as f:
    lm_data = json.load(f)

img_fn = json_fn.parent / lm_data['imagePath']
img = Image.open(img_fn)
original_size = img.size
img.thumbnail((1024, 1024))
img_scale = img.size[0] / original_size[0]
print(img_scale)
img
#%%
LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']
LINE_WIDTH = 2.0
PERP_BOX_SIZE = (20, 20)
text_attrs = dict(stroke='black',
                  stroke_width='1px',
                  fill='white',
                  style="font-size:24px; font-family:sans-serif")
import utils
from utils import calc_centroid, draw_cobb, get_circle_dict, img2b64, perp

for shape in lm_data['shapes']:
    shape['points'] = img_scale * np.array(shape['points'])

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
#%%
w, h = img.size
pixel_size = 1 / 3
logger.info('drawing')
dwg = svgwrite.Drawing('lateral.svg',
                       size=(w, h),
                       viewBox=('0 0 {} {}'.format(w, h)),
                       stroke_width=LINE_WIDTH,
                       fill='none')
logger.info('image')
dwg.add(dwg.image('data:image/jpg;base64,' + img2b64(img), size=img.size))
drawings = []
# centroids
if False:
    drawings.append(('Centroids', 'Centroids'))
    color = 'green'
    grp = dwg.add(
        dwg.g(id='Centroids', stroke=color, stroke_width=2.0, fill='none'))
    for c in vert_centroids:
        grp.add(dwg.circle(center=c.tolist(), r=2, stroke=color, fill=color))
# Thoracic kyphosis
gid = 'Thoracickyphosis'
drawings.append((gid, 'Thoracic kyphosis (p.87)'))
color = 'red'
sup = vert_sup_cnrs[2]
inf = vert_inf_cnrs[12]
grp = draw_cobb(sup, inf, dwg, gid, color, text_attrs=text_attrs)

# Mid/Lower thoracic kyphosis
gid = 'Mid/Lowerthoracickyphosis'
drawings.append((gid, 'Mid/Lower thoracic kyphosis (p.87)'))
color = 'green'
sup = vert_sup_cnrs[5]
inf = vert_inf_cnrs[12]
grp = draw_cobb(sup, inf, dwg, gid, color, text_attrs=text_attrs)

# Proximal thoracic kyphosis
gid = 'Proximalthoracickyphosis'
drawings.append((gid, 'Proximal thoracic kyphosis (p.87)'))
color = 'blue'
sup = vert_sup_cnrs[2]
inf = vert_inf_cnrs[5]
grp = draw_cobb(sup, inf, dwg, gid, color, text_attrs=text_attrs)

# Pelvic Incidence
gid = 'PelvicIncidence'
logger.info(gid)
drawings.append((gid, 'Pelvic Incidence (p.97)'))
color = 'cyan'
grp = dwg.add(dwg.g(id=gid, stroke=color, fill='none'))
circles = get_circle_dict(lm_data)
femoral_heads = np.array(circles['FemoralHead'])
grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
for femoral_head in femoral_heads:
    grp.add(
        dwg.circle(center=femoral_head[0].tolist(),
                   r=2,
                   stroke=color,
                   fill=color))
    r = np.linalg.norm(femoral_head[0] - femoral_head[1])
    grp.add(
        dwg.circle(center=femoral_head[0].tolist(),
                   r=r,
                   stroke=color,
                   fill='none'))
grp.add(dwg.path((['M'] + femoral_heads[:, 0].ravel().tolist())))
femoral_center = femoral_heads[:, 0].mean(axis=0)
grp.add(
    dwg.circle(center=femoral_center.tolist(), r=2, stroke=color, fill=color))
grp.add(dwg.path((['M'] + femoral_center.tolist() + sacrum_center.tolist())))

sacrum_width = np.linalg.norm(sacrum_tltr[1] - sacrum_tltr[0])
sacrum_perp = perp(sacrum_tltr[1] - sacrum_tltr[0])
grp.add(
    dwg.path((['M'] + sacrum_center.tolist() +
              (sacrum_center + sacrum_width * sacrum_perp).tolist())))
angle = Line(*sacrum_tltr).angle_between(Line(femoral_center, sacrum_center))
txt_angle = '{:.2f}°'.format(90 - np.rad2deg(float(angle.evalf())))
grp.add(dwg.text(txt_angle, insert=sacrum_center, **text_attrs))
# Sacral Slope
gid = 'SacralSlope'
logger.info(gid)
drawings.append((gid, 'Sacral Slope (p.99)'))
color = 'magenta'
grp = dwg.add(dwg.g(id=gid, stroke=color))
grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
sacrum_hl = sacrum_tltr[0] + sacrum_width * np.array(
    (np.sign(sacrum_tltr[1, 0] - sacrum_tltr[0, 0]), 0))
grp.add(dwg.path((['M'] + sacrum_tltr[0].tolist() + sacrum_hl.tolist())))
angle = Line(*sacrum_tltr).angle_between(Line(sacrum_tltr[0], sacrum_hl))
txt_angle = '{:.2f}°'.format(90 - np.rad2deg(float(angle.evalf())))
grp.add(dwg.text(txt_angle, insert=sacrum_center, **text_attrs))
# Pelvic Radius Angle
gid = 'PelvicRadiusAngle'
logger.info(gid)
drawings.append((gid, 'Pelvic Radius Angle (p.101)'))
color = 'blue'
grp = dwg.add(dwg.g(id=gid, stroke=color))
grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
grp.add(dwg.path((['M'] + femoral_center.tolist() + sacrum_tltr[1].tolist())))
angle = Line(*sacrum_tltr).angle_between(Line(femoral_center, sacrum_tltr[1]))
txt_angle = '{:.2f}°'.format(np.rad2deg(float(angle.evalf())))
grp.add(dwg.text(txt_angle, insert=sacrum_tltr[1], **text_attrs))
# L5 Incidence Angle
gid = 'L5IncidenceAngle'
logger.info(gid)
drawings.append((gid, 'L5 Incidence Angle (p.102)'))
color = 'red'
grp = dwg.add(dwg.g(id=gid, stroke=color))
l5_sup = vert_sup_cnrs[-1]
l5_sup_center = l5_sup.mean(axis=0)
l5_perp = perp(l5_sup[1] - l5_sup[0])
l5_width = np.linalg.norm(l5_sup[1] - l5_sup[0])
grp.add(
    dwg.path((['M'] + l5_sup_center.tolist() +
              (l5_sup_center + l5_width * l5_perp).tolist())))
grp.add(dwg.path((['M'] + vert_sup_cnrs[-1].ravel().tolist())))
grp.add(dwg.path((['M'] + femoral_center.tolist() + l5_sup_center.tolist())))
angle = Line(*l5_sup).angle_between(Line(femoral_center, l5_sup_center))
txt_angle = '{:.2f}°'.format(90 - np.rad2deg(float(angle.evalf())))
grp.add(dwg.text(txt_angle, insert=l5_sup_center, **text_attrs))
# Lumbosacral Angle
gid = 'LumbosacralAngle'
logger.info(gid)
drawings.append((gid, 'Lumbosacral Angle (p.105)'))
color = 'green'
# grp = dwg.add(dwg.g(id=gid, stroke=color))
grp = draw_cobb(vert_inf_cnrs[-1],
                sacrum_tltr,
                dwg,
                gid,
                color,
                text_attrs=text_attrs)

# save
dwg.save()
dwg
# %%
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
template = env.get_template('template.html')
t_cb = env.get_template('t_checkbox.html')
cbs = []
for name, label in drawings:
    cbs.append(t_cb.render(name=name, label=label))
with open('側面計測例.html', 'w', encoding='utf8') as f:
    f.write(template.render(svg=dwg.tostring(), checkboxes='\n'.join(cbs)))
# %%
