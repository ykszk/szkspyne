#%%
from pathlib import Path
import json
import sys
from typing import List, Union, Any
import numpy as np
import logzero
from logzero import logger
from skspatial.objects import Line, Point
from szkspyne import settings
import yaml
import svgwrite
from jinja2 import Environment, FileSystemLoader
import argparse
from szkspyne import measure_utils as utils
from szkspyne.measure_utils import calc_centroid, perp, get_circle_dict, calc_angle, Canvas

LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20)  # perpendicular sign

def lerp(a, b, t):
    if t > 1 or 1 < 0:
        raise ValueError('Invalid value for lerp: %f', t)
    return t * a + (1 - t) * b

def measure_json(lm_data: Any, height: int, config_filename: str,
                 draw_points: bool, draw_vertebral_number: bool,
                 image_type: Union[str, None], image_filename: Union[Path,
                                                                     None],
                 html_filename: Union[Path,
                                      None], measure_filename: Union[Path,
                                                                     None],
                 template_directory: str, html_title: str, hide: List[str],
                 embed_svg: bool, image_dir: Path, calc_slopes: bool):

    if not embed_svg and not str(image_filename).endswith('.svg'):
        raise RuntimeError('.svg output is required to link external svg')

    dwg = Canvas(lm_data, image_dir, height)
    kps, text_attrs, img = dwg.kps, dwg.text_attrs, dwg.img

    with open(config_filename) as f:
        color_data = yaml.safe_load(f)
    label_colors = color_data['label_colors']

    drawings = []

    if draw_points:
        drawings.append(('Points', 'Points'))
        grp = dwg.add(dwg.g(id='Points', stroke_width=LINE_WIDTH, fill='none'))
        for label, pts in kps.items():
            if label not in label_colors:
                logger.warning(
                    'No label `%s` found in defined colors in `%s`. Using black',
                    label, lm_data['filename'])
                color = '#000000'  # default to black
            else:
                color = '#' + ''.join(
                    ['{:02x}'.format(c) for c in label_colors[label]])
            for pt in pts:
                grp.add(
                    dwg.circle(center=pt,
                               r=POINT_RADIUS,
                               stroke=color,
                               fill=color))

    measures = {'flags': lm_data['flags']}
    if len(kps['TL']) != len(kps['TR']) or len(kps['BL']) != len(kps['BR']):
        raise NotImplementedError
    sacrum_tltr, vert_corners, disc_corners = utils.get_vert_and_disc_corners(
        kps)
    vert_corners_w_c7 = vert_corners.copy()
    c7_corners = vert_corners[0].copy()
    # remove C7
    vert_corners = np.delete(vert_corners, 0, axis=0)
    disc_corners = np.delete(disc_corners, 0, axis=0)
    vert_corners = vert_corners.astype(np.float64)

    logger.debug('Compute centroids')
    sacrum_center = sacrum_tltr.mean(axis=0)
    vert_centroids = np.array([calc_centroid(c) for c in vert_corners])
    c7_centroid = calc_centroid(c7_corners)
    logger.debug('Compute vecs')
    vert_sup_cnrs = vert_corners[:, :2]
    vert_sup_vecs = vert_sup_cnrs[:, 1] - vert_sup_cnrs[:, 0]
    vert_sup_vecs /= np.linalg.norm(vert_sup_vecs, axis=1, keepdims=True)
    vert_inf_cnrs = vert_corners[:, 2:]
    vert_inf_vecs = vert_inf_cnrs[:, 1] - vert_inf_cnrs[:, 0]
    vert_inf_vecs /= np.linalg.norm(vert_inf_vecs, axis=1, keepdims=True)

    if draw_vertebral_number:
        gid = 'VertebralNumber'
        drawings.append((gid, 'Vertebral Number'))
        vn_attrs = text_attrs.copy()
        vn_attrs.update(dict(stroke_width='2px'))
        grp = dwg.add(
            dwg.g(id=gid,
                    text_anchor="middle",
                    dominant_baseline="central",
                    **vn_attrs))
        for i, vc in enumerate(vert_centroids, 1):
            if i > 12:
                vn = f'L{i-12}'
            else:
                vn = f'T{i}'
            grp.add(dwg.text(vn, insert=vc))

    if image_type == 'ap':
        #%% coronal translation
        logger.debug('Coronal translation')
        coronal_translation = vert_centroids[:, 0] - sacrum_center[0]
        # Find vertebraes and calculate Cobb angles
        best_set = None
        best_score = 0
        MT_MARGIN = 3

        for mt_sup in range(MT_MARGIN, 7):
            for mt_inf in range(9, len(vert_sup_cnrs) - MT_MARGIN):
                if mt_inf - mt_sup >= 10:
                    continue
                mt_angle = calc_angle(vert_sup_cnrs[mt_sup],
                                        vert_inf_cnrs[mt_inf])
                # pt
                best_pt_score = 0
                best_pt = None
                pt_inf = mt_sup
                for pt_sup in range(0, mt_sup - 1):
                    pt_angle = calc_angle(vert_sup_cnrs[pt_sup],
                                            vert_inf_cnrs[pt_inf])
                    score = np.abs(pt_angle - mt_angle)
                    if score > best_pt_score:
                        best_pt = (pt_sup, pt_inf)
                        best_pt_score = score

                # lt
                lt_sup = mt_inf
                lt_infs = np.arange(mt_inf, len(vert_inf_cnrs))
                best_lt_score = 0
                best_lt = None
                lt_sup = mt_inf
                for lt_inf in range(lt_sup + 1, len(vert_inf_cnrs)):
                    lt_angle = calc_angle(vert_sup_cnrs[lt_sup],
                                            vert_inf_cnrs[lt_inf])
                    score = np.abs(lt_angle - mt_angle)
                    if score > best_lt_score:
                        best_lt = (lt_sup, lt_inf)
                        best_lt_score = score
                score = best_pt_score + best_lt_score
                assert best_pt is not None and best_lt is not None
                if score > best_score:
                    best_score = score
                    best_set = (mt_sup, mt_inf), best_pt, best_lt
        i_apex = np.abs(
            coronal_translation[MT_MARGIN:-MT_MARGIN]).argmax() + MT_MARGIN
        assert best_set is not None
        (mt_sup, mt_inf), (pt_sup, pt_inf), (lt_sup, lt_inf) = best_set
        measures['curve_indices'] = {
            'MT': (mt_sup, mt_inf),
            'PT': (pt_sup, pt_inf),
            'LT': (lt_sup, lt_inf)
        }

        # cobb angles
        logger.debug('Cobb angles')
        drawings += [('PT', 'PT Cobb (p.49)'), ('MT', 'MT Cobb (p.49)'),
                        ('LT', 'LT Cobb (p.49)')]
        gid = 'PT'
        visibility = 'hidden' if gid in hide else 'visible'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke='red',
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        measures[gid] = dwg.draw_cobb(vert_sup_cnrs[pt_sup],
                                        vert_inf_cnrs[pt_inf],
                                        grp,
                                        text_attrs=text_attrs)
        gid = 'MT'
        visibility = 'hidden' if gid in hide else 'visible'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke='green',
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        measures[gid] = dwg.draw_cobb(vert_sup_cnrs[mt_sup],
                                        vert_inf_cnrs[mt_inf],
                                        grp,
                                        text_attrs=text_attrs)
        if lt_sup is not None and lt_inf is not None:
            gid = 'LT'
            visibility = 'hidden' if gid in hide else 'visible'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke='blue',
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility=visibility))
            measures[gid] = dwg.draw_cobb(vert_sup_cnrs[lt_sup],
                                            vert_inf_cnrs[lt_inf],
                                            grp,
                                            text_attrs=text_attrs)
        # Major apex
        gid = 'MajorApex'
        visibility = 'hidden' if gid in hide else 'visible'
        drawings.append((gid, 'Major Apex'))
        color = 'orange'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        vert_apex = vert_corners[i_apex]
        grp.add(
            dwg.path((['M'] + [
                vc.tolist() for vc in [
                    vert_apex[0], vert_apex[1], vert_apex[3], vert_apex[2],
                    vert_apex[0]
                ]
            ])))
        for v in vert_centroids:
            grp.add(dwg.circle(center=v, r=POINT_RADIUS, fill=color))
        # apical translation
        gid = 'APT'
        visibility = 'hidden' if gid in hide else 'visible'
        drawings.append((gid, 'Apical Translation (p.53)'))
        logger.debug(drawings[-1][1])
        vert_apex = vert_centroids[i_apex]
        color = 'green'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        grp.add(
            dwg.path((['M'] + sacrum_center.tolist() + [sacrum_center[0]] +
                        [vert_corners[:, :, 1].min()])))
        grp.add(
            dwg.path((['M'] + vert_apex.tolist() + [sacrum_center[0]] +
                        [vert_centroids[i_apex, 1]])))
        grp.add(
            dwg.circle(center=vert_apex.tolist(),
                        r=POINT_RADIUS,
                        stroke=color,
                        fill=color))
        grp.add(
            dwg.text('{:.1f} pixels'.format(vert_apex[0] -
                                            sacrum_center[0]),
                        insert=vert_apex,
                        **text_attrs))
        grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
        for pos in np.concatenate((sacrum_tltr, sacrum_center[np.newaxis]),
                                    axis=0):
            grp.add(
                dwg.circle(center=pos.tolist(),
                            r=POINT_RADIUS,
                            stroke=color,
                            fill=color))

        # coronal balance
        gid = 'CoronalBalance'
        visibility = 'hidden' if gid in hide else 'visible'
        drawings.append((gid, 'Coronal Balance (p.54)'))
        logger.debug(drawings[-1][1])
        color = 'blue'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))


        grp.add(
            dwg.path((['M'] + sacrum_center.tolist() + [sacrum_center[0]] +
                        [lerp(sacrum_center[1], c7_centroid[1], .4)])))
        grp.add(
            dwg.path((['M'] + c7_centroid.tolist() + [c7_centroid[0]] +
                        [lerp(sacrum_center[1], c7_centroid[1], .6)])))
        grp.add(
            dwg.circle(center=sacrum_center.tolist(),
                        r=POINT_RADIUS,
                        stroke=color,
                        fill=color))
        grp.add(
            dwg.circle(center=c7_centroid.tolist(),
                        r=POINT_RADIUS,
                        stroke=color,
                        fill=color))
        y_cb = lerp(sacrum_center[1], c7_centroid[1], .5)
        grp.add(
            dwg.path((['M'] + [c7_centroid[0], y_cb] +
                        [sacrum_center[0], y_cb])))
        grp.add(
            dwg.text('{:.1f} pixels'.format(c7_centroid[0] -
                                            sacrum_center[0]),
                        insert=[max(c7_centroid[0], sacrum_center[0]), y_cb],
                        **text_attrs))
        measures['CoronalBalance'] = c7_centroid[0] - sacrum_center[0]
        # T1 tilt
        gid = 'T1Tilt'
        visibility = 'hidden' if gid in hide else 'visible'
        drawings.append((gid, 'T1 Tilt Angle (p.55)'))
        logger.debug(drawings[-1][1])
        color = 'blue'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))

        c7pl = np.array((c7_centroid, (c7_centroid[0], img.size[1])))
        t1_corners = vert_corners[0]
        t1_sup_vec = vert_sup_vecs[0]
        aux_vec = np.array([1, 0])
        t1_width = np.linalg.norm(t1_corners[1] - t1_corners[0])
        line_t1 = Line.from_points(vert_corners[0, 0], vert_corners[0, 1])

        aux_t1_tilt = Line.from_points(c7pl[0],
                                        c7pl[1]).intersect_line(line_t1)
        grp.add(
            dwg.path((['M'] +
                        (aux_t1_tilt - 2 * t1_width * t1_sup_vec).tolist() +
                        (aux_t1_tilt + 5 * t1_width * t1_sup_vec).tolist())))
        grp.add(
            dwg.path(
                (['M'] + (aux_t1_tilt - 2 * t1_width * aux_vec).tolist() +
                    (aux_t1_tilt + 5 * t1_width * aux_vec).tolist())))
        angle = calc_angle(
            vert_corners[0],
            ([0, aux_t1_tilt[1]], [img.size[0], aux_t1_tilt[1]]))
        angle_deg = np.rad2deg(angle)
        txt_angle = '{:.2f}°'.format(angle_deg)
        grp.add(dwg.text(txt_angle, insert=c7_centroid, **text_attrs))
        measures['T1Tilt'] = angle_deg

        if 'Clavicle' in kps and len(kps['Clavicle']) == 2:
            gid = 'ClavicleAngle'
            drawings.append((gid, 'Clavicle Angle (p.56)'))
            logger.debug(drawings[-1][1])
            color = 'green'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility='hidden' if gid in hide else 'visible'))
            assert kps['Clavicle'].shape == (2, 2)
            clavicle_kps = kps['Clavicle']
            line_clavicle = Line.from_points(clavicle_kps[0],
                                                clavicle_kps[1])
            chrl = Line.from_points(
                [0, clavicle_kps[:, 1].min()],
                [img.size[0], clavicle_kps[:, 1].min()])
            dwg.draw_line(clavicle_kps[0], clavicle_kps[1], grp)
            grp.add(
                dwg.path((['M'] + [0, clavicle_kps[:, 1].min()] +
                            [img.size[0], clavicle_kps[:, 1].min()])))
            if clavicle_kps[0, 1] != clavicle_kps[
                    1,
                    1]:  # TODO: somehow chrl.intersect_line(line_clavicle) raises ValueError of parallel
                clavicle_kps = np.array(
                    sorted(clavicle_kps, key=lambda e: e[0]))  # sort by x

                try:
                    clavicle_intersection = chrl.intersect_line(
                        line_clavicle)
                    angle = calc_angle(
                        ([0, 0], [10, 0]),  # horizontal line
                        clavicle_kps)
                except ValueError as e:  # parallel lines
                    angle = 0
                    clavicle_intersection = clavicle_kps[1]

                angle = -angle
                angle_deg = np.rad2deg(angle)
                txt_angle = '{:.1f}°'.format(angle_deg)
                grp.add(
                    dwg.text(txt_angle,
                                insert=clavicle_intersection,
                                **text_attrs))
                for pos in clavicle_kps:
                    grp.add(
                        dwg.circle(center=pos.tolist(),
                                    r=POINT_RADIUS,
                                    stroke=color,
                                    fill=color))
                measures['ClavicleAngle'] = angle_deg

        # Shoulder Height
        if 'Shoulder' in kps and len(kps['Shoulder']) == 2:
            gid = 'ShoulderHeight'
            drawings.append((gid, 'Shoulder Height (p.57)'))
            logger.debug(drawings[-1][1])
            color = 'magenta'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility='hidden' if gid in hide else 'visible'))
            shoulder_kps = kps['Shoulder']
            shoulder_kps = np.array(
                sorted(shoulder_kps, key=lambda e: e[0]))  # sort by x
            dwg.draw_vertical_dist(shoulder_kps[0], shoulder_kps[1], grp)

            for pos in shoulder_kps:
                grp.add(
                    dwg.circle(center=pos.tolist(),
                                r=POINT_RADIUS,
                                stroke=color,
                                fill=color))
            measures[gid] = shoulder_kps[0, 1] - shoulder_kps[1, 1]

        # Pelvic Obliquity
        if 'Pelvis' in kps and len(kps['Pelvis']) == 2:
            gid = 'PelvicObliquity'
            drawings.append((gid, 'Pelvic Obliquity (p.93)'))
            logger.debug(drawings[-1][1])
            color = 'cyan'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility='hidden' if gid in hide else 'visible'))
            pelvic_kps = kps['Pelvis']
            assert pelvic_kps.shape == (2, 2)
            pelvic_kps = np.array(sorted(pelvic_kps,
                                            key=lambda e: e[0]))  # sort by x
            pelvic_ref_line = Line.from_points(pelvic_kps[0],
                                                pelvic_kps[1])
            dwg.draw_line(pelvic_kps[0], pelvic_kps[1], grp)
            y_up = pelvic_kps[:, 1].min()
            grp.add(dwg.path((['M'] + [0, y_up] + [img.size[0], y_up])))
            for pos in pelvic_kps:
                grp.add(
                    dwg.circle(center=pos.tolist(),
                                r=POINT_RADIUS,
                                stroke=color,
                                fill=color))
            hline = Line.from_points((0, y_up), (img.size[0], y_up))
            if pelvic_kps[0, 1] != pelvic_kps[1, 1]:
                pelvic_cross = pelvic_ref_line.intersect_line(hline)
                angle = calc_angle(((0, y_up), (img.size[0], y_up)),
                                    pelvic_kps)
                angle = np.abs(angle)
                if pelvic_kps[0, 1] > pelvic_kps[1, 1]:
                    angle *= -1
                angle_deg = np.rad2deg(angle)
                txt_angle = '{:.1f}°'.format(angle_deg)
                grp.add(
                    dwg.text(txt_angle, insert=pelvic_cross, **text_attrs))
                grp.add(
                    dwg.rect(insert=(-LINE_WIDTH, y_up),
                                size=PERP_BOX_SIZE))
                measures[gid] = angle_deg

        # Leg Length Discrepancy
        femoral_kps = None
        if 'FemoralHead' in kps and len(kps['FemoralHead']) == 2:
            gid = 'LegLength'
            drawings.append((gid, 'Leg Length Discrepancy (p.93)'))
            logger.debug(drawings[-1][1])
            color = 'lime'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility='hidden' if gid in hide else 'visible'))
            femoral_kps = kps['FemoralHead']
            assert femoral_kps.shape == (2, 2)
            femoral_kps = np.array(sorted(femoral_kps,
                                            key=lambda e: e[0]))  # sort by x
            dwg.draw_vertical_dist(femoral_kps[0], femoral_kps[1], grp)
            for pos in femoral_kps:
                grp.add(
                    dwg.circle(center=pos.tolist(),
                                r=POINT_RADIUS,
                                stroke=color,
                                fill=color))
            measures[gid] = femoral_kps[1, 1] - femoral_kps[0, 1]

        # Sacral Obliquity
        if femoral_kps is not None:
            gid = 'SacralObliquity'
            drawings.append((gid, 'Sacral Obliquity (p.94)'))
            logger.debug(drawings[-1][1])
            color = 'hotpink'
            x_pos1, x_pos2 = femoral_kps[:, 0]
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility='hidden' if gid in hide else 'visible'))
            (_, y_pos1), (_, y_pos2) = dwg.draw_line(sacrum_tltr[0],
                                                        sacrum_tltr[1],
                                                        grp,
                                                        xrange=(x_pos1,
                                                                x_pos2))

            angle_deg = dwg.draw_cobb(
                np.array([[x_pos1, y_pos1], [x_pos2, y_pos2]]),
                femoral_kps, grp, 1, text_attrs)
            for pos in np.concatenate((femoral_kps, sacrum_tltr), axis=0):
                grp.add(
                    dwg.circle(center=pos.tolist(),
                                r=POINT_RADIUS,
                                stroke=color,
                                fill=color))
            measures[gid] = angle_deg
    else:  # lateral
        # Thoracic kyphosis
        gid = 'ThoracicKyphosis'
        drawings.append((gid, 'Thoracic kyphosis (p.87)'))
        color = 'red'
        visibility = 'hidden' if gid in hide else 'visible'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        sup = vert_sup_cnrs[1]
        inf = vert_inf_cnrs[11]
        angle_deg = dwg.draw_cobb(sup, inf, grp, text_attrs=text_attrs)
        measures[gid] = angle_deg

        # Mid/Lower thoracic kyphosis
        gid = 'Mid/LowerThoracicKyphosis'
        drawings.append((gid, 'Mid/Lower thoracic kyphosis (p.87)'))
        color = 'green'
        visibility = 'hidden' if gid in hide else 'visible'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        sup = vert_sup_cnrs[4]
        inf = vert_inf_cnrs[11]
        angle_deg = dwg.draw_cobb(sup, inf, grp, text_attrs=text_attrs)

        measures[gid] = angle_deg

        # Proximal thoracic kyphosis
        gid = 'ProximalThoracicKyphosis'
        drawings.append((gid, 'Proximal thoracic kyphosis (p.87)'))
        color = 'blue'
        visibility = 'hidden' if gid in hide else 'visible'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        sup = vert_sup_cnrs[1]
        inf = vert_inf_cnrs[4]
        angle_deg = dwg.draw_cobb(sup, inf, grp, text_attrs=text_attrs)

        measures[gid] = angle_deg

        # Lumbar lordosis (T12/S)
        gid = 'LumbarLordosis'
        drawings.append((gid, 'Lumbar lordosis (T12/S) '))
        color = 'blue'
        visibility = 'hidden' if gid in hide else 'visible'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=visibility))
        sup = vert_sup_cnrs[11]
        inf = sacrum_tltr
        angle_deg = dwg.draw_cobb(sup, inf, grp, text_attrs=text_attrs)

        measures[gid] = angle_deg

        # Sacral Slope
        gid = 'SacralSlope'
        logger.debug(gid)
        drawings.append((gid, 'Sacral Slope (p.99)'))
        color = 'magenta'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    visibility='hidden' if gid in hide else 'visible'))
        grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
        sacrum_width = np.linalg.norm(sacrum_tltr[1] - sacrum_tltr[0])
        sacrum_hl = sacrum_tltr[0] + sacrum_width * np.array(
            (np.sign(sacrum_tltr[1, 0] - sacrum_tltr[0, 0]), 0))
        grp.add(
            dwg.path(
                (['M'] + sacrum_tltr[0].tolist() + sacrum_hl.tolist())))
        angle = np.abs(calc_angle(sacrum_tltr,
                                    (sacrum_tltr[0], sacrum_hl)))
        angle_deg = np.rad2deg(angle)
        txt_angle = '{:.2f}°'.format(angle_deg)
        grp.add(dwg.text(txt_angle, insert=sacrum_center, **text_attrs))
        measures['SacralSlope'] = angle_deg

        # C7-SVA
        gid = 'C7SVA'
        logger.debug(gid)
        drawings.append((gid, 'Sagittal Balance (p.67)'))
        color = 'red'
        grp = dwg.add(
            dwg.g(id=gid,
                    stroke=color,
                    visibility='hidden' if gid in hide else 'visible'))

        sacrum_tr = sacrum_tltr[1]
        grp.add(
            dwg.path((['M'] + sacrum_tr.tolist() + [sacrum_tr[0]] +
                        [lerp(sacrum_tr[1], c7_centroid[1], .7)])))
        grp.add(
            dwg.path((['M'] + c7_centroid.tolist() + [c7_centroid[0]] +
                        [lerp(sacrum_tr[1], c7_centroid[1], .9)])))
        grp.add(
            dwg.circle(center=sacrum_tr.tolist(),
                        r=POINT_RADIUS,
                        stroke=color,
                        fill=color))
        grp.add(
            dwg.circle(center=c7_centroid.tolist(),
                        r=POINT_RADIUS,
                        stroke=color,
                        fill=color))
        y_cb = lerp(sacrum_tr[1], c7_centroid[1], .8)
        balance = sacrum_tr[0] - c7_centroid[0]
        grp.add(
            dwg.path(
                (['M'] + [c7_centroid[0], y_cb] + [sacrum_tr[0], y_cb])))
        grp.add(
            dwg.text('{:.1f} pixels'.format(balance),
                        insert=[max(c7_centroid[0], sacrum_tr[0]), y_cb],
                        **text_attrs))
        measures[gid] = balance

        # Pelvic Incidence
        circles = get_circle_dict(lm_data)
        if 'FemoralHead' in circles and len(circles['FemoralHead']) == 2:
            gid = 'PelvicIncidence'
            logger.debug(gid)
            drawings.append((gid, 'Pelvic Incidence (p.97)'))
            color = 'cyan'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        fill='none',
                        visibility='hidden' if gid in hide else 'visible'))
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
            grp.add(
                dwg.path((['M'] + femoral_heads[:, 0].ravel().tolist())))
            femoral_center = femoral_heads[:, 0].mean(axis=0)
            grp.add(
                dwg.circle(center=femoral_center.tolist(),
                            r=2,
                            stroke=color,
                            fill=color))
            grp.add(
                dwg.path((['M'] + femoral_center.tolist() +
                            sacrum_center.tolist())))

            sacrum_perp = perp(sacrum_tltr[1] - sacrum_tltr[0])
            grp.add(
                dwg.path((['M'] + sacrum_center.tolist() +
                            (sacrum_center +
                            sacrum_width * sacrum_perp).tolist())))
            angle = np.abs(
                calc_angle(sacrum_tltr, (femoral_center, sacrum_center)))
            angle_deg = 90 - np.rad2deg(angle)
            txt_angle = '{:.2f}°'.format(angle_deg)
            grp.add(dwg.text(txt_angle, insert=sacrum_center,
                                **text_attrs))
            measures['PelvicIncidence'] = angle_deg

            # Pelvic Radius Angle
            gid = 'PelvicRadiusAngle'
            logger.debug(gid)
            drawings.append((gid, 'Pelvic Radius Angle (p.101)'))
            color = 'blue'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        visibility='hidden' if gid in hide else 'visible'))
            grp.add(dwg.path((['M'] + sacrum_tltr.ravel().tolist())))
            grp.add(
                dwg.path((['M'] + femoral_center.tolist() +
                            sacrum_tltr[1].tolist())))
            angle = np.abs(
                calc_angle(sacrum_tltr, (femoral_center, sacrum_tltr[1])))
            angle_deg = np.rad2deg(angle)
            txt_angle = '{:.2f}°'.format(angle_deg)
            grp.add(
                dwg.text(txt_angle, insert=sacrum_tltr[1], **text_attrs))
            measures['PelvicRadiusAngle'] = angle_deg
            # L5 Incidence Angle
            gid = 'L5IncidenceAngle'
            logger.debug(gid)
            drawings.append((gid, 'L5 Incidence Angle (p.102)'))
            color = 'red'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        visibility='hidden' if gid in hide else 'visible'))
            l5_sup = vert_sup_cnrs[-1]
            l5_sup_center = l5_sup.mean(axis=0)
            l5_perp = perp(l5_sup[1] - l5_sup[0])
            l5_width = np.linalg.norm(l5_sup[1] - l5_sup[0])
            grp.add(
                dwg.path((['M'] + l5_sup_center.tolist() +
                            (l5_sup_center + l5_width * l5_perp).tolist())))
            grp.add(dwg.path((['M'] + vert_sup_cnrs[-1].ravel().tolist())))
            grp.add(
                dwg.path((['M'] + femoral_center.tolist() +
                            l5_sup_center.tolist())))
            angle = np.abs(
                calc_angle(l5_sup, (femoral_center, l5_sup_center)))
            angle_deg = 90 - np.rad2deg(angle)
            txt_angle = '{:.2f}°'.format(angle_deg)
            grp.add(dwg.text(txt_angle, insert=l5_sup_center,
                                **text_attrs))
            measures['L5IncidenceAngle'] = angle_deg
            # Lumbosacral Angle
            gid = 'LumbosacralAngle'
            logger.debug(gid)
            drawings.append((gid, 'Lumbosacral Angle (p.105)'))
            color = 'green'
            visibility = 'hidden' if gid in hide else 'visible'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility=visibility))
            angle_deg = dwg.draw_cobb(vert_inf_cnrs[-1],
                                        sacrum_tltr,
                                        grp,
                                        text_attrs=text_attrs)
            measures[gid] = angle_deg

    if calc_slopes:
        grps = {}
        for ul, name, color in [('upper', 'Upper slope', 'red'),
                                ('lower', 'Lower slope', 'blue')]:
            gid = f'{ul}_slope'
            grp = dwg.add(
                dwg.g(id=gid,
                        stroke=color,
                        visibility='hidden' if gid in hide else 'visible'))
            grps[gid] = grp
            drawings.append((gid, name))
        for i, corners in enumerate(vert_corners_w_c7, 1):
            for pair, ul, color in ((corners[:2], 'upper', 'red'),
                                    (corners[2:], 'lower', 'blue')):
                name = f'V{i}_{ul}_slope'
                gid = f'{ul}_slope'
                v = pair[1] - pair[0]
                angle = -np.arctan2(
                    v[1], v[0]
                )  # negation because y axis in image is upside down
                angle_deg = np.rad2deg(angle)
                txt_angle = '{:.2f}°'.format(angle_deg)
                measures[name] = angle_deg
                grp = grps[gid]
                grp.add(dwg.path((['M'] + pair.ravel().tolist())))
                grp.add(dwg.text(txt_angle, insert=pair[0], **text_attrs))
            name = f'V{i}_center'
            center = corners.mean(axis=0)
            measures[name + '_x'] = center[0]
            measures[name + '_y'] = center[1]
        circles = get_circle_dict(lm_data)
        measures['sacral_tl_x'] = sacrum_tltr[0][0]
        measures['sacral_tl_y'] = sacrum_tltr[0][1]
        measures['sacral_tr_x'] = sacrum_tltr[1][0]
        measures['sacral_tr_y'] = sacrum_tltr[1][1]
        for i, pts in enumerate(circles.get('FemoralHead', []), 1):
            name = f'femoral_head_{i}'
            measures[name + '_x'] = pts[0][0]
            measures[name + '_y'] = pts[0][1]

    if image_filename:
        filename = Path(image_filename)
        if filename.suffix == '.svg':
            dwg.saveas(filename)
        elif filename.suffix == '.png':
            raise NotImplementedError
        else:
            raise NotImplementedError

    if measure_filename:
        with open(measure_filename, 'w') as f:
            json.dump(measures, f, ensure_ascii=False)

    if html_filename:
        env = Environment(
            loader=FileSystemLoader(template_directory, encoding='utf8'))
        template = env.get_template('template.html')
        t_cb = env.get_template('t_checkbox.html')

        cbs = []
        for name, label in drawings:
            cbs.append(
                t_cb.render(name=name,
                            label=label,
                            checked='' if name in hide else 'checked'))
        assert image_filename is not None
        svg_tag = dwg.tostring(
        ) if embed_svg else '<object id="svgfile" type="image/svg+xml" data="{}"></object>'.format(
            Path(image_filename).name)
        with open(html_filename, 'w', encoding='utf8') as f:
            f.write(
                template.render(
                    svg=svg_tag,
                    svg_container='document' if embed_svg else
                    'document.getElementById("svgfile").contentDocument',
                    checkboxes='\n'.join(cbs),
                    title=html_title))
    return measures


def main():
    parser = argparse.ArgumentParser(description='Measure and draw.')
    parser.add_argument('-c',
                        '--config',
                        help='Config yaml filename',
                        required=True)
    parser.add_argument(
        '-t',
        '--template',
        help='template directory. default: ${SCRIPT_DIR}/template',
        default=Path(sys.argv[0]).parent / 'template')
    parser.add_argument('--title',
                        help='html title',
                        default='Scoliosis Measuaraments')
    parser.add_argument('--image_type',
                        help='Image type',
                        choices=['ap', 'lateral'],
                        default='ap')
    parser.add_argument('--hide',
                        help='Hide specified drawings',
                        nargs='*',
                        default=[])
    parser.add_argument('--parent_dir',
                        help='Override parent directory of imagePath')

    parser.add_argument('--height',
                        help='Output image height',
                        type=int,
                        default=768)
    parser.add_argument('--points', help='Draw points', action='store_true')
    parser.add_argument('--slopes',
                        help='Add slopes to measuremens json',
                        action='store_true')
    parser.add_argument('--numbers',
                        help='Draw vertebral numbers',
                        action='store_true')
    parser.add_argument('--verbose', help='Print logs', action='store_true')
    parser.add_argument('--external_svg',
                        help='Use external svg file for html output.',
                        action='store_true')

    subparsers = parser.add_subparsers()
    sub_single = subparsers.add_parser('single',
                                       help="measure and draw from given file")
    sub_single.add_argument('input',
                            help='Input json filename',
                            metavar='<input>',
                            type=Path)
    sub_single.add_argument('--measure', help='Save measurements json')
    sub_single.add_argument('--html', help='Save html')
    sub_single.add_argument('--image', help='Save image (svg or png)')

    def command_single(args):
        if args.parent_dir:
            image_dir = Path(args.parent_dir)
        else:
            image_dir = Path(args.input).parent
        with open(args.input) as f:
            lm_data = json.load(f)
        measure_json(lm_data, args.height, args.config, args.points,
                     args.numbers, args.image_type, args.image, args.html,
                     args.measure, args.template, args.title, args.hide,
                     not args.external_svg, image_dir, args.slopes)

    sub_single.set_defaults(handler=command_single)

    sub_batch = subparsers.add_parser(
        'batch', help="measure and draw from given directory")
    sub_batch.add_argument('input', help='Input json directory')
    sub_batch.add_argument('output', help='Output directory')
    sub_batch.add_argument('--measure',
                           help='Save measurements json',
                           action='store_true')
    sub_batch.add_argument('--ndjson',
                           help='Save all measurements as a ndjson file')
    sub_batch.add_argument('--html', help='Save html', action='store_true')
    sub_batch.add_argument('--image',
                           help='Save image',
                           metavar='ext',
                           choices=['.svg', '.png'])
    sub_batch.add_argument(
        '--workers',
        help='The number of concurrent workers. default:%(default)s',
        type=int,
        default=4)

    def command_batch(args):
        if not args.html and not args.image and not args.measure:
            print(
                'At least one output (html, image, or measure) argument is required.'
            )
            sys.exit(1)
        outdir = Path(args.output)
        from joblib import Parallel, delayed
        import tqdm

        def do(fn: Path):
            measure = (outdir /
                       fn.name).with_suffix('.json') if args.measure else None
            image = (outdir /
                     fn.name).with_suffix(args.image) if args.image else None
            html = (outdir /
                    fn.name).with_suffix('.html') if args.html else None
            if args.parent_dir:
                image_dir = Path(args.parent_dir)
            else:
                image_dir = fn.parent
            with open(fn) as f:
                lm_data = json.load(f)
                lm_data['filename'] = fn.stem
            return measure_json(lm_data, args.height, args.config, args.points,
                                args.numbers, args.image_type, image, html,
                                measure, args.template,
                                '{} - {}'.format(fn.stem,
                                                 args.title), args.hide,
                                not args.external_svg, image_dir, args.slopes)

        def do_ndjson(fn: Path, lm_data):
            measure = (outdir /
                       lm_data['filename']).with_suffix('.json') if args.measure else None
            image = (outdir /
                     lm_data['filename']).with_suffix(args.image) if args.image else None
            html = (outdir /
                    lm_data['filename']).with_suffix('.html') if args.html else None
            if args.parent_dir:
                image_dir = Path(args.parent_dir)
            else:
                image_dir = fn.parent
            return measure_json(lm_data, args.height, args.config, args.points,
                                args.numbers, args.image_type, image, html,
                                measure, args.template,
                                '{} - {}'.format(fn.stem,
                                                 args.title), args.hide,
                                not args.external_svg, image_dir, args.slopes)

        if Path(args.input).is_dir():
            json_filenames = sorted(Path(args.input).glob('*.json'))
            all_measures = Parallel(n_jobs=args.workers)(
                delayed(do)(fn) for fn in tqdm.tqdm(json_filenames))
        else:
            with open(args.input) as f:
                ndjson = [json.loads(line) for line in f]
            indir = Path(args.input).parent
            json_filenames = [indir / data['filename'] for data in ndjson]
            all_measures = Parallel(n_jobs=args.workers)(
                delayed(do_ndjson)(Path(args.input), data)
                for data in tqdm.tqdm(ndjson))
        assert all_measures is not None
        if args.ndjson:
            with open(args.ndjson, 'w') as f:
                for fn, ms in zip(json_filenames, all_measures):
                    ms['filename'] = fn.stem
                    f.write(json.dumps(ms, ensure_ascii=False))
                    f.write('\n')
        return 0

    sub_batch.set_defaults(handler=command_batch)

    args = parser.parse_args()

    logzero.loglevel(30)
    if args.verbose:
        logzero.loglevel(20)

    if not hasattr(args, 'handler'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)

    return (args.handler(args))


if __name__ == "__main__":
    sys.exit(main())