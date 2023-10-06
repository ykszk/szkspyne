import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import logzero
import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader
from logzero import logger
from PIL import Image
from skspatial.objects import Line

import szkspyne.utils.measure as utils
from szkspyne.utils.measure import (
    CurveIndices,
    SpinalDrawing,
    calc_angle,
    find_apex,
    get_circle_dict,
    lerp,
    perp,
)


LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20)  # perpendicular sign


def load_line_colors():
    return utils.read_default_line_colors()


def load_label_colors(filename):
    with open(filename) as f:
        color_data = yaml.safe_load(f)
    label_colors = color_data['label_colors']
    return label_colors


def measure_json(
    lm_data: Any,
    height: int,
    label_colors: Dict[str, str],
    line_colors: Dict[str, str],
    draw_points: bool,
    draw_vertebral_number: bool,
    image_type: Optional[str],
    html_filename: Optional[Path],
    template_directory: str,
    html_title: str,
    hide: List[str],
    image_dir: Path,
    calc_slopes: bool,
    imagedb: Optional[str],
):
    dwg = SpinalDrawing(lm_data, image_dir, height)
    kps, text_attrs, image_size = dwg.vertebrae.kps, dwg.text_attrs, dwg.size

    d_visibility = defaultdict(lambda: 'visible')
    d_visibility.update({gid: 'hidden' for gid in hide})

    drawings = []

    if imagedb:
        with sqlite3.connect(imagedb) as conn:
            cursor = conn.cursor()
            filename = lm_data['filename'] + '.png'
            cursor.execute("SELECT binary FROM image WHERE name = ?", (filename,))
            binary_data = cursor.fetchone()
        if binary_data:
            binary_data = binary_data[0]
            gid = 'Heatmap'
            grp = dwg.add(dwg.g(id=gid, class_=gid, visibility='hidden'))
            file_jpgdata = BytesIO(binary_data)
            img = Image.open(file_jpgdata)
            img = img.resize((int(round(img.size[0] * height / img.size[1])), height))
            grp.add(dwg.image('data:image/png;base64,' + utils.img2b64(img, format='png'), size=img.size))

            drawings.append((gid, gid))

    if draw_points:
        gid = 'Points'
        drawings.append((gid, 'Points'))
        grp = dwg.add(dwg.g(id=gid, class_=gid, stroke_width=LINE_WIDTH))
        for label, pts in kps.items():
            if label not in label_colors:
                logger.warning(
                    'No label `%s` found in defined colors in `%s`. Using black', label, lm_data['filename']
                )
                color = '#000000'  # default to black
            else:
                color = '#' + ''.join(['{:02x}'.format(c) for c in label_colors[label]])
            g_ptx = dwg.g(id=f'Points-{label}', stroke=color, fill=color)
            for pt in pts:
                g_ptx.add(dwg.circle(center=pt, r=POINT_RADIUS))
            grp.add(g_ptx)

    measures: Dict[str, Any] = {}
    all_alt_measures = []
    measure_result = {'flags': lm_data['flags'], 'measures': measures, 'alt_measures': all_alt_measures}

    sacrum_tltr, vert_corners = dwg.vertebrae.sacrum_tltr, dwg.vertebrae.vert_corners
    c7_centroid = dwg.vertebrae.c7_centroid

    sacrum_center, vert_centroids = dwg.vertebrae.sacrum_center, dwg.vertebrae.vert_centroids
    vert_sup_cnrs, vert_sup_vecs, vert_sup_vecs, vert_inf_cnrs = (
        dwg.vertebrae.vert_sup_cnrs,
        dwg.vertebrae.vert_sup_vecs,
        dwg.vertebrae.vert_sup_vecs,
        dwg.vertebrae.vert_inf_cnrs,
    )

    if draw_vertebral_number:
        gid = 'VertebralNumber'
        drawings.append((gid, 'Vertebral Number'))
        vn_attrs = text_attrs.copy()
        vn_attrs.update(dict(stroke_width='2px'))
        grp = dwg.add(dwg.g(id=gid, class_=gid, text_anchor="middle", dominant_baseline="central", **vn_attrs))
        for i, vc in enumerate(vert_centroids, 1):
            if i > 12:
                vn = f'L{i-12}'
            else:
                vn = f'T{i}'
            grp.add(dwg.text(vn, insert=vc))

    layer_control = None

    if image_type == 'ap':
        for i, d_curve_indices in enumerate(lm_data['curve_indices']):
            alt_measures = {'name': d_curve_indices['name']}
            all_alt_measures.append(alt_measures)
            curve_indices = CurveIndices.from_dict(d_curve_indices['indices'])
            alt_measures['curve_indices'] = curve_indices.to_dict()
            display = 'inline' if i == 0 else 'none'
            layer = dwg.add(dwg.g(id=f'layer_{i}', class_=f'layer layer_{i}', style=f'display:{display}'))

            logger.debug('Cobb angles')
            apexes: Dict[str, int] = d_curve_indices.get('apices', {})
            for gid, disp, sup_inf_pair in [
                ('MT', 'MT Cobb (p.49)', curve_indices.mt),
                ('TLL', 'TL/L Cobb (p.49)', curve_indices.tll),
                ('PT', 'PT Cobb (p.49)', curve_indices.pt),
            ]:
                if sup_inf_pair is not None:
                    sup, inf = sup_inf_pair
                    apex = apexes.get(gid, find_apex(vert_corners, vert_centroids, sup, inf))
                    apexes[gid] = apex
                    grp = layer.add(
                        dwg.g(
                            id=gid,
                            class_=gid,
                            stroke=line_colors[gid],
                            stroke_width=LINE_WIDTH,
                            fill='none',
                            visibility=d_visibility[gid],
                        )
                    )
                    angle = dwg.draw_cobb(
                        vert_sup_cnrs[sup], vert_inf_cnrs[inf], grp, title=gid, text_attrs=text_attrs
                    )
                    alt_measures[gid] = angle
                    if i == 0:
                        drawings.append((gid, disp))
            # Curve apex
            if len(apexes) > 0:
                gid = 'CurveApex'
                if i == 0:
                    drawings.append((gid, 'Curve Apex(es)'))
                color = line_colors[gid]
                grp = layer.add(
                    dwg.g(
                        id=gid,
                        class_=gid,
                        stroke=color,
                        stroke_width=LINE_WIDTH,
                        fill='none',
                        visibility=d_visibility[gid],
                    )
                )
                for apex in apexes.values():
                    vert_apex = vert_corners[apex]
                    grp.add(
                        dwg.polygon(
                            points=[vc.tolist() for vc in [vert_apex[0], vert_apex[1], vert_apex[3], vert_apex[2]]]
                        )
                    )
                for v in vert_centroids:
                    grp.add(dwg.circle(center=v, r=POINT_RADIUS, fill=color))
        for key in all_alt_measures[0].keys():
            if key == 'name':
                continue
            measures[key] = all_alt_measures[0][key]
        if len(lm_data['curve_indices']) > 1:
            layer_control = [(f'layer_{i}', data['name']) for i, data in enumerate(lm_data['curve_indices'])]

        # apical translation
        if False:
            gid = 'APT'
            drawings.append((gid, 'Apical Translation (p.53)'))
            logger.debug(drawings[-1][1])
            vert_apex = vert_centroids[i_apex]
            color = line_colors[gid]
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            grp.add(dwg.line(sacrum_center.tolist(), [sacrum_center[0], vert_corners[:, :, 1].min()]))
            grp.add(dwg.line(vert_apex.tolist(), [sacrum_center[0], vert_centroids[i_apex, 1]]))
            grp.add(dwg.circle(center=vert_apex.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
            text_elm = dwg.text('{:.1f} px'.format(vert_apex[0] - sacrum_center[0]), insert=vert_apex, **text_attrs)
            text_elm.set_desc(title=gid)
            grp.add(text_elm)
            grp.add(dwg.line(sacrum_tltr[0], sacrum_tltr[1]))
            for pos in np.concatenate((sacrum_tltr, sacrum_center[np.newaxis]), axis=0):
                grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))

        # coronal balance
        gid = 'CoronalBalance'
        drawings.append((gid, 'Coronal Balance (p.54)'))
        logger.debug(drawings[-1][1])
        color = line_colors[gid]
        grp = dwg.add(
            dwg.g(id=gid, class_=gid, stroke=color, stroke_width=LINE_WIDTH, fill='none', visibility=d_visibility[gid])
        )

        grp.add(dwg.line(sacrum_center.tolist(), [sacrum_center[0], lerp(sacrum_center[1], c7_centroid[1], 0.4)]))
        grp.add(dwg.line(c7_centroid.tolist(), [c7_centroid[0], lerp(sacrum_center[1], c7_centroid[1], 0.6)]))
        grp.add(dwg.circle(center=sacrum_center.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
        grp.add(dwg.circle(center=c7_centroid.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
        y_cb = lerp(sacrum_center[1], c7_centroid[1], 0.5)
        grp.add(dwg.line([c7_centroid[0], y_cb], [sacrum_center[0], y_cb]))
        text_elm = dwg.text(
            '{:.1f} px'.format(c7_centroid[0] - sacrum_center[0]),
            insert=[max(c7_centroid[0], sacrum_center[0]), y_cb],
            **text_attrs,
        )
        text_elm.set_desc(title=gid)
        grp.add(text_elm)
        measures['CoronalBalance'] = c7_centroid[0] - sacrum_center[0]
        # T1 tilt
        gid = 'T1Tilt'
        drawings.append((gid, 'T1 Tilt Angle (p.55)'))
        logger.debug(drawings[-1][1])
        color = line_colors[gid]
        grp = dwg.add(
            dwg.g(id=gid, class_=gid, stroke=color, stroke_width=LINE_WIDTH, fill='none', visibility=d_visibility[gid])
        )

        c7pl = np.array((c7_centroid, (c7_centroid[0], image_size[1])))
        t1_corners = vert_corners[0]
        t1_sup_vec = vert_sup_vecs[0]
        aux_vec = np.array([1, 0])
        t1_width = np.linalg.norm(t1_corners[1] - t1_corners[0])
        line_t1 = Line.from_points(vert_corners[0, 0], vert_corners[0, 1])

        aux_t1_tilt = Line.from_points(c7pl[0], c7pl[1]).intersect_line(line_t1)
        grp.add(
            dwg.line(
                (aux_t1_tilt - 2 * t1_width * t1_sup_vec).tolist(), (aux_t1_tilt + 5 * t1_width * t1_sup_vec).tolist()
            )
        )
        grp.add(
            dwg.line((aux_t1_tilt - 2 * t1_width * aux_vec).tolist(), (aux_t1_tilt + 5 * t1_width * aux_vec).tolist())
        )
        angle = calc_angle(vert_corners[0], ([0, aux_t1_tilt[1]], [image_size[0], aux_t1_tilt[1]]))
        angle_deg = np.rad2deg(angle)
        txt_angle = '{:.2f}°'.format(angle_deg)
        text_elm = dwg.text(txt_angle, insert=c7_centroid, **text_attrs)
        text_elm.set_desc(title=gid)
        grp.add(text_elm)
        measures['T1Tilt'] = angle_deg

        if 'Clavicle' in kps and len(kps['Clavicle']) == 2:
            gid = 'ClavicleAngle'
            drawings.append((gid, 'Clavicle Angle (p.56)'))
            logger.debug(drawings[-1][1])
            color = line_colors[gid]
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            assert kps['Clavicle'].shape == (2, 2)
            clavicle_kps = kps['Clavicle']
            line_clavicle = Line.from_points(clavicle_kps[0], clavicle_kps[1])
            chrl = Line.from_points([0, clavicle_kps[:, 1].min()], [image_size[0], clavicle_kps[:, 1].min()])
            dwg.draw_line(clavicle_kps[0], clavicle_kps[1], grp)
            grp.add(dwg.line([0, clavicle_kps[:, 1].min()], [image_size[0], clavicle_kps[:, 1].min()]))
            if (
                clavicle_kps[0, 1] != clavicle_kps[1, 1]
            ):  # TODO: somehow chrl.intersect_line(line_clavicle) raises ValueError of parallel
                clavicle_kps = np.array(sorted(clavicle_kps, key=lambda e: e[0]))  # sort by x

                try:
                    clavicle_intersection = chrl.intersect_line(line_clavicle)
                    angle = calc_angle(([0, 0], [10, 0]), clavicle_kps)  # horizontal line
                except ValueError as e:  # parallel lines
                    angle = 0
                    clavicle_intersection = clavicle_kps[1]

                angle = -angle
                angle_deg = np.rad2deg(angle)
                txt_angle = '{:.1f}°'.format(angle_deg)
                text_elm = dwg.text(txt_angle, insert=clavicle_intersection, **text_attrs)
                text_elm.set_desc(title=gid)
                grp.add(text_elm)
                for pos in clavicle_kps:
                    grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
                measures['ClavicleAngle'] = angle_deg

        # Shoulder Height
        if 'Shoulder' in kps and len(kps['Shoulder']) == 2:
            gid = 'ShoulderHeight'
            drawings.append((gid, 'Shoulder Height (p.57)'))
            logger.debug(drawings[-1][1])
            color = line_colors[gid]
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            shoulder_kps = kps['Shoulder']
            shoulder_kps = np.array(sorted(shoulder_kps, key=lambda e: e[0]))  # sort by x
            dwg.draw_vertical_dist(shoulder_kps[0], shoulder_kps[1], grp, title=gid)

            for pos in shoulder_kps:
                grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
            measures[gid] = shoulder_kps[0, 1] - shoulder_kps[1, 1]

        # Pelvic Obliquity
        if 'Pelvis' in kps and len(kps['Pelvis']) == 2:
            gid = 'PelvicObliquity'
            drawings.append((gid, 'Pelvic Obliquity (p.93)'))
            logger.debug(drawings[-1][1])
            color = line_colors[gid]
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            pelvic_kps = kps['Pelvis']
            assert pelvic_kps.shape == (2, 2)
            pelvic_kps = np.array(sorted(pelvic_kps, key=lambda e: e[0]))  # sort by x
            pelvic_ref_line = Line.from_points(pelvic_kps[0], pelvic_kps[1])
            dwg.draw_line(pelvic_kps[0], pelvic_kps[1], grp)
            y_up = pelvic_kps[:, 1].min()
            grp.add(dwg.line([0, y_up], [image_size[0], y_up]))
            for pos in pelvic_kps:
                grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
            hline = Line.from_points((0, y_up), (image_size[0], y_up))
            if pelvic_kps[0, 1] != pelvic_kps[1, 1]:
                pelvic_cross = pelvic_ref_line.intersect_line(hline)
                angle = calc_angle(((0, y_up), (image_size[0], y_up)), pelvic_kps)
                angle = np.abs(angle)
                if pelvic_kps[0, 1] > pelvic_kps[1, 1]:
                    angle *= -1
                angle_deg = np.rad2deg(angle)
                txt_angle = '{:.1f}°'.format(angle_deg)
                text_elm = dwg.text(txt_angle, insert=pelvic_cross, **text_attrs)
                text_elm.set_desc(title=gid)
                grp.add(text_elm)
                grp.add(dwg.rect(insert=(-LINE_WIDTH, y_up), size=PERP_BOX_SIZE))
                measures[gid] = angle_deg

        # Leg Length Discrepancy
        femoral_kps = None
        if 'FemoralHead' in kps and len(kps['FemoralHead']) == 2:
            gid = 'LegLength'
            drawings.append((gid, 'Leg Length Discrepancy (p.93)'))
            logger.debug(drawings[-1][1])
            color = line_colors[gid]
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            femoral_kps = kps['FemoralHead']
            assert femoral_kps.shape == (2, 2)
            femoral_kps = np.array(sorted(femoral_kps, key=lambda e: e[0]))  # sort by x
            dwg.draw_vertical_dist(femoral_kps[0], femoral_kps[1], grp, title=gid)
            for pos in femoral_kps:
                grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
            measures[gid] = femoral_kps[1, 1] - femoral_kps[0, 1]

        # Sacral Obliquity
        if femoral_kps is not None:
            gid = 'SacralObliquity'
            drawings.append((gid, 'Sacral Obliquity (p.94)'))
            logger.debug(drawings[-1][1])
            color = line_colors[gid]
            x_pos1, x_pos2 = femoral_kps[:, 0]
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            (_, y_pos1), (_, y_pos2) = dwg.draw_line(sacrum_tltr[0], sacrum_tltr[1], grp, xrange=(x_pos1, x_pos2))

            angle_deg = dwg.draw_cobb(
                np.array([[x_pos1, y_pos1], [x_pos2, y_pos2]]), femoral_kps, grp, 1, title=gid, text_attrs=text_attrs
            )
            for pos in np.concatenate((femoral_kps, sacrum_tltr), axis=0):
                grp.add(dwg.circle(center=pos.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
            measures[gid] = angle_deg
    else:  # lateral
        # Thoracic kyphosis
        gid = 'ThoracicKyphosis'
        drawings.append((gid, 'Thoracic kyphosis (p.87)'))
        color = line_colors[gid]
        grp = dwg.add(
            dwg.g(id=gid, class_=gid, stroke=color, stroke_width=LINE_WIDTH, fill='none', visibility=d_visibility[gid])
        )
        sup = vert_sup_cnrs[1]
        inf = vert_inf_cnrs[11]
        angle_deg = dwg.draw_cobb(sup, inf, grp, title=gid, text_attrs=text_attrs)
        measures[gid] = angle_deg

        # Mid/Lower thoracic kyphosis
        gid = 'Mid/LowerThoracicKyphosis'
        drawings.append((gid, 'Mid/Lower thoracic kyphosis (p.87)'))
        color = line_colors[gid]
        grp = dwg.add(
            dwg.g(id=gid, class_=gid, stroke=color, stroke_width=LINE_WIDTH, fill='none', visibility=d_visibility[gid])
        )
        sup = vert_sup_cnrs[4]
        inf = vert_inf_cnrs[11]
        angle_deg = dwg.draw_cobb(sup, inf, grp, title=gid, text_attrs=text_attrs)

        measures[gid] = angle_deg

        # Proximal thoracic kyphosis
        gid = 'ProximalThoracicKyphosis'
        drawings.append((gid, 'Proximal thoracic kyphosis (p.87)'))
        color = line_colors[gid]
        grp = dwg.add(
            dwg.g(id=gid, class_=gid, stroke=color, stroke_width=LINE_WIDTH, fill='none', visibility=d_visibility[gid])
        )
        sup = vert_sup_cnrs[1]
        inf = vert_inf_cnrs[4]
        angle_deg = dwg.draw_cobb(sup, inf, grp, title=gid, text_attrs=text_attrs)

        measures[gid] = angle_deg

        # Lumbar lordosis (T12/S)
        gid = 'LumbarLordosis'
        drawings.append((gid, 'Lumbar lordosis (T12/S) '))
        color = line_colors[gid]
        grp = dwg.add(
            dwg.g(id=gid, class_=gid, stroke=color, stroke_width=LINE_WIDTH, fill='none', visibility=d_visibility[gid])
        )
        sup = vert_sup_cnrs[11]
        inf = sacrum_tltr
        angle_deg = dwg.draw_cobb(sup, inf, grp, title=gid, text_attrs=text_attrs)

        measures[gid] = angle_deg

        # Sacral Slope
        gid = 'SacralSlope'
        logger.debug(gid)
        drawings.append((gid, 'Sacral Slope (p.99)'))
        color = line_colors[gid]
        grp = dwg.add(dwg.g(id=gid, class_=gid, stroke=color, visibility=d_visibility[gid]))
        grp.add(dwg.line(sacrum_tltr[0].tolist(), sacrum_tltr[1].tolist()))
        sacrum_width = np.linalg.norm(sacrum_tltr[1] - sacrum_tltr[0])
        sacrum_hl = sacrum_tltr[0] + sacrum_width * np.array((np.sign(sacrum_tltr[1, 0] - sacrum_tltr[0, 0]), 0))
        grp.add(dwg.line(sacrum_tltr[0].tolist(), sacrum_hl.tolist()))
        angle = np.abs(calc_angle(sacrum_tltr, (sacrum_tltr[0], sacrum_hl)))
        angle_deg = np.rad2deg(angle)
        txt_angle = '{:.2f}°'.format(angle_deg)
        text_elm = dwg.text(txt_angle, insert=sacrum_center, **text_attrs)
        text_elm.set_desc(title=gid)
        grp.add(text_elm)
        measures['SacralSlope'] = angle_deg

        # C7-SVA
        gid = 'C7SVA'
        logger.debug(gid)
        drawings.append((gid, 'Sagittal Balance (p.67)'))
        color = line_colors[gid]
        grp = dwg.add(dwg.g(id=gid, class_=gid, stroke=color, stroke_width=LINE_WIDTH, visibility=d_visibility[gid]))

        sacrum_tr = sacrum_tltr[1]
        grp.add(dwg.line(sacrum_tr.tolist(), [sacrum_tr[0]] + [lerp(sacrum_tr[1], c7_centroid[1], 0.7)]))
        grp.add(dwg.line(c7_centroid.tolist(), [c7_centroid[0]] + [lerp(sacrum_tr[1], c7_centroid[1], 0.9)]))
        grp.add(dwg.circle(center=sacrum_tr.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
        grp.add(dwg.circle(center=c7_centroid.tolist(), r=POINT_RADIUS, stroke=color, fill=color))
        y_cb = lerp(sacrum_tr[1], c7_centroid[1], 0.8)
        balance = sacrum_tr[0] - c7_centroid[0]
        grp.add(dwg.line([c7_centroid[0], y_cb], [sacrum_tr[0], y_cb]))
        text_elm = dwg.text(
            '{:.1f} px'.format(balance), insert=[max(c7_centroid[0], sacrum_tr[0]), y_cb], **text_attrs
        )
        text_elm.set_desc(title=gid)
        grp.add(text_elm)
        measures[gid] = balance

        # Pelvic Incidence
        circles = get_circle_dict(lm_data)
        if 'FemoralHead' in circles and len(circles['FemoralHead']) == 2:
            gid = 'PelvicIncidence'
            logger.debug(gid)
            drawings.append((gid, 'Pelvic Incidence (p.97)'))
            color = line_colors[gid]
            grp = dwg.add(dwg.g(id=gid, class_=gid, stroke=color, fill='none', visibility=d_visibility[gid]))
            femoral_heads = np.array(circles['FemoralHead'])
            grp.add(dwg.line(sacrum_tltr[0], sacrum_tltr[1]))
            for femoral_head in femoral_heads:
                grp.add(dwg.circle(center=femoral_head[0].tolist(), r=2, stroke=color, fill=color))
                r = np.linalg.norm(femoral_head[0] - femoral_head[1])
                grp.add(dwg.circle(center=femoral_head[0].tolist(), r=r, stroke=color, fill='none'))
            grp.add(dwg.line(femoral_heads[0, 0], femoral_heads[1, 0]))
            femoral_center = femoral_heads[:, 0].mean(axis=0)
            grp.add(dwg.circle(center=femoral_center.tolist(), r=2, stroke=color, fill=color))
            grp.add(dwg.line(femoral_center.tolist(), sacrum_center.tolist()))

            sacrum_perp = perp(sacrum_tltr[1] - sacrum_tltr[0])
            grp.add(dwg.line(sacrum_center, (sacrum_center + sacrum_width * sacrum_perp).tolist()))
            angle = np.abs(calc_angle(sacrum_tltr, (femoral_center, sacrum_center)))
            angle_deg = 90 - np.rad2deg(angle)
            txt_angle = '{:.2f}°'.format(angle_deg)
            text_elm = dwg.text(txt_angle, insert=sacrum_center, **text_attrs)
            text_elm.set_desc(title=gid)
            grp.add(text_elm)
            measures['PelvicIncidence'] = angle_deg

            # Pelvic Radius Angle
            gid = 'PelvicRadiusAngle'
            logger.debug(gid)
            drawings.append((gid, 'Pelvic Radius Angle (p.101)'))
            color = line_colors[gid]
            grp = dwg.add(dwg.g(id=gid, class_=gid, stroke=color, visibility=d_visibility[gid]))
            grp.add(dwg.line(sacrum_tltr[0].tolist(), sacrum_tltr[1].tolist()))
            grp.add(dwg.line(femoral_center.tolist(), sacrum_tltr[1].tolist()))
            angle = np.abs(calc_angle(sacrum_tltr, (femoral_center, sacrum_tltr[1])))
            angle_deg = np.rad2deg(angle)
            txt_angle = '{:.2f}°'.format(angle_deg)
            text_elm = dwg.text(txt_angle, insert=sacrum_tltr[1], **text_attrs)
            text_elm.set_desc(title=gid)
            grp.add(text_elm)
            measures['PelvicRadiusAngle'] = angle_deg
            # L5 Incidence Angle
            gid = 'L5IncidenceAngle'
            logger.debug(gid)
            drawings.append((gid, 'L5 Incidence Angle (p.102)'))
            color = line_colors[gid]
            grp = dwg.add(dwg.g(id=gid, class_=gid, stroke=color, visibility=d_visibility[gid]))
            l5_sup = vert_sup_cnrs[-1]
            l5_sup_center = l5_sup.mean(axis=0)
            l5_perp = perp(l5_sup[1] - l5_sup[0])
            l5_width = np.linalg.norm(l5_sup[1] - l5_sup[0])
            grp.add(dwg.line(l5_sup_center.tolist(), (l5_sup_center + l5_width * l5_perp).tolist()))
            grp.add(dwg.line(vert_sup_cnrs[-1][0], vert_sup_cnrs[-1][1]))
            grp.add(dwg.line(femoral_center.tolist(), l5_sup_center.tolist()))
            angle = np.abs(calc_angle(l5_sup, (femoral_center, l5_sup_center)))
            angle_deg = 90 - np.rad2deg(angle)
            txt_angle = '{:.2f}°'.format(angle_deg)
            text_elm = dwg.text(txt_angle, insert=l5_sup_center, **text_attrs)
            text_elm.set_desc(title=gid)
            grp.add(text_elm)
            measures['L5IncidenceAngle'] = angle_deg
            # Lumbosacral Angle
            gid = 'LumbosacralAngle'
            logger.debug(gid)
            drawings.append((gid, 'Lumbosacral Angle (p.105)'))
            color = line_colors[gid]
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke=color,
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            angle_deg = dwg.draw_cobb(vert_inf_cnrs[-1], sacrum_tltr, grp, title=gid, text_attrs=text_attrs)
            measures[gid] = angle_deg

    if calc_slopes:
        grps = {}
        for ul, name in [('upper', 'Upper slope'), ('lower', 'Lower slope')]:
            gid = f'{ul}_slope'
            color = line_colors[gid]
            grp = dwg.add(dwg.g(id=gid, class_=gid, stroke=color, visibility=d_visibility[gid]))
            grps[gid] = grp
            drawings.append((gid, name))
        for i, corners in enumerate(dwg.vertebrae.vert_corners_w_c7, 1):
            for pair, ul in ((corners[:2], 'upper'), (corners[2:], 'lower')):
                name = f'V{i}_{ul}_slope'
                gid = f'{ul}_slope'
                v = pair[1] - pair[0]
                angle = -np.arctan2(v[1], v[0])  # negation because y axis in image is upside down
                angle_deg = np.rad2deg(angle)
                txt_angle = '{:.2f}°'.format(angle_deg)
                measures[name] = angle_deg
                grp = grps[gid]
                grp.add(dwg.line(pair[0], pair[1]))
                text_elm = dwg.text(txt_angle, insert=pair[0], **text_attrs)
                text_elm.set_desc(title=gid)
                grp.add(text_elm)
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

    if html_filename:
        env = Environment(loader=FileSystemLoader(template_directory, encoding='utf8'))
        template = env.get_template('template.html')
        t_cb = env.get_template('t_checkbox.html')

        cbs = []
        for name, label in drawings:
            cbs.append(t_cb.render(name=name, label=label, checked='' if name in hide else 'checked'))
        if layer_control:
            t_lc = env.get_template('switch_layer.html')
            lc = t_lc.render(layers=layer_control)
            cbs.append(lc)
        svg_tag = dwg.tostring()
        with open(html_filename, 'w', encoding='utf8') as f:
            f.write(
                template.render(svg=svg_tag, svg_container='document', checkboxes='\n'.join(cbs), title=html_title)
            )
    return measure_result


def main():
    parser = argparse.ArgumentParser(description='Measure and draw.')
    parser.add_argument('-c', '--config', help='Config yaml filename', required=True)
    parser.add_argument(
        '-t',
        '--template',
        help='template directory. default: ${SCRIPT_DIR}/template',
        default=Path(sys.argv[0]).parent / 'template',
    )
    parser.add_argument('--title', help='html title', default='Scoliosis Measuaraments')
    parser.add_argument('--image_type', help='Image type', choices=['ap', 'lateral'], default='ap')
    parser.add_argument('--imagedb', help='image database filename for heatmap overlay')
    parser.add_argument('--hide', help='Hide specified drawings', nargs='*', default=[])
    parser.add_argument('--parent_dir', help='Override parent directory of imagePath')

    parser.add_argument('--height', help='Output image height', type=int, default=768)
    parser.add_argument('--points', help='Draw points', action='store_true')
    parser.add_argument('--slopes', help='Add slopes to measuremens json', action='store_true')
    parser.add_argument('--numbers', help='Draw vertebral numbers', action='store_true')
    parser.add_argument('--verbose', help='Print logs', action='store_true')

    subparsers = parser.add_subparsers()
    sub_single = subparsers.add_parser('single', help="measure and draw from given file")
    sub_single.add_argument('input', help='Input json filename', metavar='<input>', type=Path)
    sub_single.add_argument('--measure', help='Save measurements json')
    sub_single.add_argument('--html', help='Save html')

    def command_single(args):
        label_colors = load_label_colors(args.config)
        line_colors = load_line_colors()
        if args.parent_dir:
            image_dir = Path(args.parent_dir)
        else:
            image_dir = Path(args.input).parent
        with open(args.input) as f:
            lm_data = json.load(f)
        measures = measure_json(
            lm_data,
            args.height,
            label_colors,
            line_colors,
            args.points,
            args.numbers,
            args.image_type,
            args.html,
            args.template,
            args.title,
            args.hide,
            image_dir,
            args.slopes,
            args.imagedb,
        )
        if args.measure:
            with open(args.measure, 'w') as f:
                json.dump(measures, f, ensure_ascii=False)

    sub_single.set_defaults(handler=command_single)

    sub_batch = subparsers.add_parser('batch', help="measure and draw from given directory")
    sub_batch.add_argument('input', help='Input json directory or ndjson filename or "-" for stdin in ndjson format')
    sub_batch.add_argument('output', help='Output directory')
    sub_batch.add_argument('--measure', help='Save measurements json', action='store_true')
    sub_batch.add_argument('--ndjson', help='Save all measurements as a ndjson file')
    sub_batch.add_argument('--html', help='Save html', action='store_true')
    sub_batch.add_argument('--image', help='Save image', metavar='ext', choices=['.svg', '.png'])
    sub_batch.add_argument(
        '--workers', help='The number of concurrent workers. Use all available cores by default', type=int, default=0
    )

    def command_batch(args):
        if not args.html and not args.image and not args.measure:
            print('At least one output (html, image, or measure) argument is required.')
            sys.exit(1)
        if args.measure and args.ndjson:
            logger.warning('Specifying both `measure` and `ndjson is somewhat redundant. Consider using onlye one.')
        outdir = Path(args.output)
        label_colors = load_label_colors(args.config)
        line_colors = load_line_colors()
        if args.workers <= 0:
            import os

            args.workers = os.cpu_count()
            logger.info("Use %d cores", args.workers)
        import tqdm
        from joblib import Parallel, delayed

        def do(fn: Path):
            with open(fn) as f:
                lm_data = json.load(f)
                lm_data['filename'] = fn.stem
            return do_ndjson(fn, lm_data)

        def do_ndjson(fn: Path, lm_data):
            stem = re.sub(r"\.json$", "", Path(lm_data['filename']).name)
            measure_filename = outdir / (stem + '.json') if args.measure else None
            html = outdir / (stem + '.html') if args.html else None
            if args.parent_dir:
                image_dir = Path(args.parent_dir)
            else:
                image_dir = fn.parent
            measures = measure_json(
                lm_data,
                args.height,
                label_colors,
                line_colors,
                args.points,
                args.numbers,
                args.image_type,
                html,
                args.template,
                '{} - {}'.format(stem, args.title),
                args.hide,
                image_dir,
                args.slopes,
                args.imagedb,
            )
            if measure_filename:
                with open(measure_filename, 'w') as f:
                    json.dump(measures, f, ensure_ascii=False)
            return measures

        if Path(args.input).is_dir():
            json_filenames = sorted(Path(args.input).glob('*.json'))
            all_measures = Parallel(n_jobs=args.workers)(delayed(do)(fn) for fn in tqdm.tqdm(json_filenames))
        else:
            if args.input == '-':
                ndjson = [json.loads(line) for line in sys.stdin]
            else:
                with open(args.input) as f:
                    ndjson = [json.loads(line) for line in f]
            indir = Path(args.input).parent
            json_filenames = [indir / data['filename'] for data in ndjson]
            all_measures = Parallel(n_jobs=args.workers)(
                delayed(do_ndjson)(Path(args.input), data) for data in tqdm.tqdm(ndjson)
            )
        assert all_measures is not None
        if args.ndjson:
            with open(args.ndjson, 'w') as f:
                for fn, ms in zip(json_filenames, all_measures):
                    ms['filename'] = re.sub(r"\.json$", "", fn.name)  # not using `fn.stem` to avoid stripping too much
                    f.write(json.dumps(ms, ensure_ascii=False, separators=(',', ':')))
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

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
