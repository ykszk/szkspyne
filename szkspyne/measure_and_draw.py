#%%
from pathlib import Path
import json

from PIL import Image
import numpy as np
from logzero import logger
import sympy
from sympy.geometry import Point, Line, Polygon, Segment, Circle, intersection
from sympy import solve
from sympy.abc import x, y
import yaml
import svgwrite
from jinja2 import Environment, FileSystemLoader
import argparse

LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20) # perpendicular sign

parser = argparse.ArgumentParser(
    description='Measure and draw.')
parser.add_argument('input', help='Input json filename', metavar='<input>')
parser.add_argument('output', help='Output html filename', metavar='<output>')
parser.add_argument('-c', '--config',
                    help='Config yaml filename',required=True)
parser.add_argument('-t', '--template',
                    help='template directory', default='.')
parser.add_argument('--title',
                    help='html title', default='Scoliosis Measuaraments')
parser.add_argument('--image',
                    help='Save image (svg or png)')
parser.add_argument('--image_type',
                    help='Image type',choices=['ap','lateral'],default='ap')
parser.add_argument('--height',
                    help='Output image height', type=int, default=768)
parser.add_argument('--points',
                    help='Draw points',
                    action='store_true')

args = parser.parse_args()
json_fn = Path(args.input)
with open(json_fn, encoding='utf8') as f:
    lm_data = json.load(f)

img_fn = str(json_fn.parent / lm_data['imagePath']).replace('\\','/')
img = Image.open(img_fn)
original_image_shape = img.size

HEIGHT = args.height
img.thumbnail((HEIGHT,HEIGHT))
scale = HEIGHT / original_image_shape[1]

with open(args.config) as f:
    color_data = yaml.safe_load(f)
label_colors = color_data['label_colors']

for shape in lm_data['shapes']:
    shape['points'] = [[scale*e for e in p] for p in shape['points']]

LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']
import measure_utils as utils
from measure_utils import calc_centroid, draw_cobb, perp, get_circle_dict, calc_cobb

points = utils.get_point_dict(lm_data)
kps = utils.sort_kps(points)
n_vertebrae = len(kps['BR'])
for l in LABEL_NAMES: # remove C7
    kps[l] = np.delete(kps[l], 0, axis=0)
sacrum_tltr, vert_corners, disc_corners = utils.get_vert_and_disc_corners(kps)

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

w, h = img.size
pixel_size = 1
image_size=(w * pixel_size, h * pixel_size)
logger.info('drawing')
dwg = svgwrite.Drawing('output.svg',
                    size=image_size,
                    viewBox=('0 0 {} {}'.format(w, h)))
logger.info('image')
dwg.add(dwg.image('data:image/jpg;base64,' + utils.img2b64(img), size=img.size))
drawings = []

text_attrs = dict(stroke='black',
                stroke_width='1px',
                fill='white',
                style="font-size:24px; font-family:sans-serif")

if args.points:
    drawings.append(('Points', 'Points'))
    grp = dwg.add(dwg.g(id='Points', stroke_width=2.0, fill='none'))
    for label, pts in points.items():
        color = '#' + ''.join(['{:02x}'.format(c) for c in label_colors[label]])
        for pt in pts:
            grp.add(dwg.circle(center=pt, r=POINT_RADIUS, stroke=color, fill=color))

if args.image_type=='ap':
    #%% coronal translation
    logger.info('Coronal translation')
    coronal_translation = vert_centroids[:, 0] - sacrum_center[0]
    # Find vertebraes and calculate Cobb angles
    best_set = None
    best_score = 0
    for mt_sup in range(2, len(vert_sup_cnrs)-2):
        for mt_inf in range(mt_sup+1, len(vert_sup_cnrs)-2):
            mt_angle = calc_cobb(vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf])
            # find pt            
            pt_inf = mt_sup
            pt_sups = np.arange(0, mt_sup)
            pt_angles = np.abs([calc_cobb(vert_sup_cnrs[pt_sup], vert_inf_cnrs[pt_inf]) for pt_sup in pt_sups])
            pt_sup = pt_sups[np.argmax(pt_angles)]
            # find lt
            lt_sup = mt_inf
            lt_infs = np.arange(mt_inf, len(vert_inf_cnrs))
            lt_angles = np.abs([calc_cobb(vert_sup_cnrs[lt_sup], vert_inf_cnrs[lt_inf]) for lt_inf in lt_infs])
            lt_inf = lt_infs[np.argmax(lt_angles)]
            score = np.abs(mt_angle) + pt_angles.max() + lt_angles.max()
            print(mt_sup, mt_inf, pt_sup, lt_inf, score)
            if score > best_score:
                best_score = score
                best_set = mt_sup, mt_inf, pt_sup, lt_inf
    i_apex = np.abs(coronal_translation[2:-2]).argmax() + 2
    assert best_set is not None
    mt_sup, mt_inf, pt_sup, lt_inf = best_set

    # cobb angles
    logger.info('Cobb angles')
    drawings += [('PT', 'PT Cobb (p.49)'), ('MT', 'MT Cobb (p.49)'),
                ('LT', 'LT Cobb (p.49)')]
    draw_cobb(vert_sup_cnrs[pt_sup], vert_inf_cnrs[mt_sup], dwg, 'PT', 'red', text_attrs=text_attrs)
    draw_cobb(vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf], dwg, 'MT', 'green', text_attrs=text_attrs)
    draw_cobb(vert_sup_cnrs[mt_inf], vert_inf_cnrs[lt_inf], dwg, 'LT',
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
    if 'Clavicle' in kps and len(kps['Clavicle'])==2:
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
    if 'Shoulder' in kps and len(kps['Shoulder']) >= 2:
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

    # Pelvic Obliquity
    if 'Pelvis' in kps and len(kps['Pelvis']) >= 2:
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
    femoral_kps = None
    if 'FemoralHead' in kps and len(kps['FemoralHead']) >= 2:
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
    if femoral_kps is not None:
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
else: # lateral
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

    # Sacral Slope
    gid = 'SacralSlope'
    logger.info(gid)
    drawings.append((gid, 'Sacral Slope (p.99)'))
    color = 'magenta'
    grp = dwg.add(dwg.g(id=gid, stroke=color))
    grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
    sacrum_width = np.linalg.norm(sacrum_tltr[1] - sacrum_tltr[0])
    sacrum_hl = sacrum_tltr[0] + sacrum_width * np.array(
        (np.sign(sacrum_tltr[1, 0] - sacrum_tltr[0, 0]), 0))
    grp.add(dwg.path((['M'] + sacrum_tltr[0].tolist() + sacrum_hl.tolist())))
    angle = Line(*sacrum_tltr).angle_between(Line(sacrum_tltr[0], sacrum_hl))
    txt_angle = '{:.2f}°'.format(90 - np.rad2deg(float(angle.evalf())))
    grp.add(dwg.text(txt_angle, insert=sacrum_center, **text_attrs))


    # Pelvic Incidence
    circles = get_circle_dict(lm_data)
    if 'FemoralHeads' in circles and len(circles['FemoralHeads'])>=2:
        gid = 'PelvicIncidence'
        logger.info(gid)
        drawings.append((gid, 'Pelvic Incidence (p.97)'))
        color = 'cyan'
        grp = dwg.add(dwg.g(id=gid, stroke=color, fill='none'))
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

        sacrum_perp = perp(sacrum_tltr[1] - sacrum_tltr[0])
        grp.add(
            dwg.path((['M'] + sacrum_center.tolist() +
                    (sacrum_center + sacrum_width * sacrum_perp).tolist())))
        angle = Line(*sacrum_tltr).angle_between(Line(femoral_center, sacrum_center))
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

if args.image:
    filename = Path(args.image)
    if filename.suffix=='.svg':
        dwg.saveas(filename)
    elif filename.suffix=='.png':
        from svg import Parser, Rasterizer
        svg = Parser.parse(dwg.tostring())
        rast = Rasterizer()
        buff = rast.rasterize(svg, svg.width, svg.height)
        mes = Image.frombytes('RGBA', (svg.width, svg.height), buff)
        img = img.convert('RGBA')
        img.alpha_composite(mes)
        img = img.convert('RGB')
        img.save(filename)
    else:
        raise NotImplementedError

env = Environment(loader=FileSystemLoader(args.template, encoding='utf8'))
template = env.get_template('template.html')
t_cb = env.get_template('t_checkbox.html')

cbs = []
for name, label in drawings:
    cbs.append(t_cb.render(name=name, label=label))
with open(args.output, 'w', encoding='utf8') as f:
    f.write(template.render(svg=dwg.tostring(), checkboxes='\n'.join(cbs), title=args.title))
