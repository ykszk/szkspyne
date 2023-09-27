# %%
import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple

import logzero
import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader
from logzero import logger
from PIL import Image
from skspatial.objects import Line, LineSegment

import szkspyne.utils.measure as utils
from szkspyne.utils.measure import (
    SpinalDrawing,
    calc_angle,
    get_circle_dict,
    lerp,
    perp,
)


LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20)  # perpendicular sign


def is_crossing(l1: LineSegment, l2: LineSegment) -> bool:
    try:
        l1.intersect_line_segment(l2)
        return True
    except ValueError:
        return False


def measure_json(
    lm_data: Any,
    height: int,
    config_filename: str,
    draw_points: bool,
    draw_vertebral_number: bool,
    image_type: Optional[str],
    html_filename: Optional[Path],
    measure_filename: Optional[Path],
    template_directory: str,
    html_title: str,
    hide: List[str],
    image_dir: Path,
    calc_slopes: bool,
    curve_algo: str,
    imagedb: Optional[str],
):
    dwg = SpinalDrawing(lm_data, image_dir, height)
    kps, text_attrs, image_size = dwg.kps, dwg.text_attrs, dwg.size

    with open(config_filename) as f:
        color_data = yaml.safe_load(f)
    label_colors = color_data['label_colors']

    line_colors = utils.read_default_line_colors()
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

    measures = {'flags': lm_data['flags']}
    if len(kps['TL']) != len(kps['TR']) or len(kps['BL']) != len(kps['BR']):
        raise ValueError('Invalid set of number of points for TL, TR, BL and BR')
    sacrum_tltr, vert_corners, disc_corners = dwg.sacrum_tltr, dwg.vert_corners, dwg.disc_corners
    vert_corners_w_c7, c7_corners, c7_centroid = dwg.vert_corners_w_c7, dwg.c7_corners, dwg.c7_centroid

    sacrum_center, vert_centroids, disc_centroids = dwg.sacrum_center, dwg.vert_centroids, dwg.disc_centroids
    vert_sup_cnrs, vert_sup_vecs, vert_sup_vecs, vert_inf_cnrs = (
        dwg.vert_sup_cnrs,
        dwg.vert_sup_vecs,
        dwg.vert_sup_vecs,
        dwg.vert_inf_cnrs,
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

    if image_type == 'ap':
        # %% coronal translation
        logger.debug('Coronal translation')
        coronal_translation = vert_centroids[:, 0] - sacrum_center[0]
        # Find vertebraes and calculate Cobb angles
        best_set = None
        best_score = 0
        MT_MARGIN = 3

        def find_apex(sup, inf):
            # if include_discs:
            #     vert_disc_corners = np.concatenate([
            #         vert_corners[sup:(inf + 1)],
            #         disc_corners[sup:inf]
            #     ],
            #                                         axis=0)
            #     vert_disc_centroids = np.concatenate([
            #         vert_centroids[sup:(inf + 1)],
            #         disc_centroids[sup:inf]
            #     ],
            #                                             axis=0)
            # else:

            # maximum translation
            sup_line = Line.from_points(vert_corners[sup][0], vert_corners[sup][1])
            inf_line = Line.from_points(vert_corners[inf][0], vert_corners[inf][1])

            if True:
                vert_disc_corners = vert_corners[sup : (inf + 1)]
                vert_disc_centroids = vert_centroids[sup : (inf + 1)]
            try:
                p_intersect = sup_line.intersect_line(inf_line)
            except ValueError:  # parallel lines
                apex = (sup + inf) // 2  # TODO: implement better parallel-line handling
                return apex, vert_disc_corners[apex - sup]

            line_sup2inf = Line.from_points(vert_centroids[inf], vert_centroids[sup])
            # apex, _ = max(
            #     [(i, abs(line_sup2inf.distance_point(c)))
            #         for i, c in enumerate(vert_disc_centroids)],
            #     key=lambda e: e[1])
            apex, _ = max(
                [(i, abs(p_intersect[0] - c[0])) for i, c in enumerate(vert_disc_centroids)], key=lambda e: e[1]
            )
            apex_corner = vert_disc_corners[apex]
            return sup + apex, apex_corner

        if curve_algo == 'max_diffs':
            for mt_sup in range(MT_MARGIN, 7):
                for mt_inf in range(9, len(vert_sup_cnrs) - MT_MARGIN):
                    if mt_inf - mt_sup >= 10:
                        continue
                    mt_angle = calc_angle(vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf])
                    # pt
                    best_pt_score = 0
                    best_pt = None
                    pt_inf = mt_sup
                    for pt_sup in range(0, mt_sup - 1):
                        pt_angle = calc_angle(vert_sup_cnrs[pt_sup], vert_inf_cnrs[pt_inf])
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
                        lt_angle = calc_angle(vert_sup_cnrs[lt_sup], vert_inf_cnrs[lt_inf])
                        score = np.abs(lt_angle - mt_angle)
                        if score > best_lt_score:
                            best_lt = (lt_sup, lt_inf)
                            best_lt_score = score
                    score = best_pt_score + best_lt_score
                    assert best_pt is not None and best_lt is not None
                    if score > best_score:
                        best_score = score
                        best_set = (mt_sup, mt_inf), best_pt, best_lt
            assert best_set is not None
            (mt_sup, mt_inf), (pt_sup, pt_inf), (lt_sup, lt_inf) = best_set
            apexes = [np.abs(coronal_translation[MT_MARGIN:-MT_MARGIN]).argmax() + MT_MARGIN]
            apex_corners = [vert_corners[a] for a in apexes]
        elif curve_algo == 'fujimori':
            APEX_MARGIN = 2
            diff_mat = (
                coronal_translation[APEX_MARGIN:-APEX_MARGIN, np.newaxis]
                - coronal_translation[np.newaxis, APEX_MARGIN:-APEX_MARGIN]
            )
            max_pair: Tuple[int] = np.unravel_index(diff_mat.argmax(), diff_mat.shape)
            upper_apex = int(min(*max_pair) + APEX_MARGIN)
            lower_apex = int(max(*max_pair) + APEX_MARGIN)
            if vert_centroids[upper_apex][0] > vert_centroids[lower_apex][0]:
                upper_arg, lower_arg = np.argmin, np.argmax
            else:
                upper_arg, lower_arg = np.argmax, np.argmin
            mt_angles = []
            for mt_sup in range(upper_apex - 1):
                for mt_inf in range(upper_apex + 1, lower_apex - 1):
                    mt_angle = calc_angle(vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf])
                    mt_angles.append((mt_sup, mt_inf, mt_angle))
            mt_set = mt_angles[upper_arg([e[-1] for e in mt_angles])]
            mt_sup, mt_inf = mt_set[0], mt_set[1]

            lt_sup = mt_inf
            lt_angles = []
            for lt_inf in range(lower_apex + 1, len(vert_corners)):
                lt_angle = calc_angle(vert_sup_cnrs[lt_sup], vert_inf_cnrs[lt_inf])
                lt_angles.append((lt_sup, lt_inf, lt_angle))
            lt_set = lt_angles[lower_arg([e[-1] for e in lt_angles])]
            lt_sup, lt_inf = lt_set[0], lt_set[1]
            pt_sup, pt_inf = None, None
            apex_corners = [vert_corners[upper_apex], vert_corners[lower_apex]]
        elif curve_algo == 'no-cross':
            # no-cross: max diff with constraint
            # No PT
            n_verts = len(vert_corners)
            candidates = []

            for mt_sup in range(n_verts):
                for mt_inf in range(mt_sup + 2, 13):  # n_verts-2):
                    # check curve
                    sup2inf = LineSegment(vert_centroids[mt_inf], vert_centroids[mt_sup])
                    prev2inf = LineSegment(vert_centroids[mt_inf - 1], vert_centroids[mt_inf - 2])
                    if mt_inf - mt_sup > 3 and is_crossing(sup2inf, prev2inf):
                        break
                    lt_sup = mt_inf
                    mt_angle = calc_angle(vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf])
                    for lt_inf in range(lt_sup + 2, n_verts):
                        # check curve
                        sup2inf = LineSegment(vert_centroids[lt_inf], vert_centroids[lt_sup])
                        prev2inf = LineSegment(vert_centroids[lt_inf - 1], vert_centroids[lt_inf - 2])
                        if lt_inf - lt_sup > 3 and is_crossing(sup2inf, prev2inf):
                            break
                        lt_angle = calc_angle(vert_sup_cnrs[lt_sup], vert_inf_cnrs[lt_inf])
                        score = mt_angle - lt_angle
                        candidates.append((score, mt_sup, mt_inf, lt_sup, lt_inf))
            if len(candidates) == 0:
                raise RuntimeError('No candidate was found for %s', lm_data['imagePath'])
            scores = np.array([e[0] for e in candidates])
            abs_scores = np.abs(scores)
            best_set = candidates[np.argmax(abs_scores)]
            score, mt_sup, mt_inf, lt_sup, lt_inf = best_set
            pt_sup, pt_inf = None, None
            mt_apex, mt_apex_corner = find_apex(mt_sup, mt_inf)
            lt_apex, lt_apex_corner = find_apex(lt_sup, lt_inf)
            if n_verts - lt_inf >= 2 and lt_apex < 11:
                lt_angle = calc_angle(vert_sup_cnrs[lt_sup], vert_inf_cnrs[lt_inf])
                sup = lt_inf
                candidates = []
                for inf in range(sup + 2, n_verts):
                    angle = calc_angle(vert_sup_cnrs[sup], vert_inf_cnrs[inf])
                    if angle * lt_angle < 0:
                        candidates.append((inf, abs(angle)))
                if candidates:
                    inf, angle = max(candidates, key=lambda e: e[1])
                    pt_sup, pt_inf = mt_sup, mt_inf
                    mt_sup, mt_inf = lt_sup, lt_inf
                    lt_sup = mt_inf
                    lt_inf = inf
            if pt_sup is None and mt_sup >= 2:
                mt_angle = calc_angle(vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf])
                inf = mt_sup
                candidates = []
                for sup in range(0, inf - 1):
                    angle = calc_angle(vert_sup_cnrs[sup], vert_inf_cnrs[inf])
                    if angle * mt_angle < 0:
                        candidates.append((sup, abs(angle)))
                if candidates:
                    pt_sup, _ = max(candidates, key=lambda e: e[1])
                    pt_inf = inf

            # re-compute apexes
            mt_apex, mt_apex_corner = find_apex(mt_sup, mt_inf)
            lt_apex, lt_apex_corner = find_apex(lt_sup, lt_inf)
            apex_corners = [mt_apex_corner, lt_apex_corner]
            if pt_sup is not None:
                pt_apex, pt_apex_corner = find_apex(pt_sup, pt_inf)
                apex_corners = [pt_apex_corner] + apex_corners
        elif curve_algo == 'no-cross2':
            # no-cross: max diff with constraint
            n_verts = len(vert_corners)

            sup_inf, scores = [], []
            for sup in range(n_verts):
                for inf in range(sup + 2, n_verts):
                    # check curve
                    sup2inf = LineSegment(vert_centroids[inf], vert_centroids[sup])
                    prev2inf = LineSegment(vert_centroids[inf - 1], vert_centroids[inf - 2])
                    if inf - sup > 3 and is_crossing(sup2inf, prev2inf):
                        continue
                    angle = calc_angle(vert_sup_cnrs[sup], vert_inf_cnrs[inf])
                    scores.append(abs(angle))
                    sup_inf.append((sup, inf))
            if len(scores) == 0:
                raise RuntimeError('No candidate was found for %s', lm_data['imagePath'])
            major_sup, major_inf = sup_inf[np.argmax(scores)]
            # score, mt_sup, mt_inf, lt_sup, lt_inf = best_set
            # pt_sup, pt_inf = None, None
            major_apex, _ = find_apex(major_sup, major_inf)
            sup_inf, scores = [], []
            if major_apex <= 10:  # mt is major
                mt_sup, mt_inf = major_sup, major_inf
                lt_sup = mt_inf
                for lt_inf in range(lt_sup + 2, n_verts):
                    # check curve
                    sup2inf = LineSegment(vert_centroids[lt_inf], vert_centroids[lt_sup])
                    prev2inf = LineSegment(vert_centroids[lt_inf - 1], vert_centroids[lt_inf - 2])
                    if lt_inf - lt_sup > 3 and is_crossing(sup2inf, prev2inf):
                        continue
                    lt_angle = calc_angle(vert_sup_cnrs[lt_sup], vert_inf_cnrs[lt_inf])
                    scores.append(abs(lt_angle))
                    sup_inf.append((lt_sup, lt_inf))
                if len(scores) == 0:
                    raise RuntimeError('No candidate was found for %s. %d, %d', lm_data['imagePath'], mt_sup, mt_inf)
                else:
                    lt_sup, lt_inf = sup_inf[np.argmax(scores)]
            else:  # tll is major
                lt_sup, lt_inf = major_sup, major_inf
                mt_inf = lt_sup
                for mt_sup in range(max(mt_inf - 2, 0)):
                    # check curve
                    sup2inf = LineSegment(vert_centroids[mt_inf], vert_centroids[mt_sup])
                    prev2inf = LineSegment(vert_centroids[mt_inf - 1], vert_centroids[mt_inf - 2])
                    if mt_inf - mt_sup > 3 and is_crossing(sup2inf, prev2inf):
                        continue
                    mt_angle = calc_angle(vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf])
                    scores.append(abs(mt_angle))
                    sup_inf.append((mt_sup, mt_inf))
                if len(scores) == 0:
                    raise RuntimeError('No candidate was found for %s. %d, %d', lm_data['imagePath'], lt_sup, lt_inf)
                else:
                    mt_sup, mt_inf = sup_inf[np.argmax(scores)]

            # re-compute apexes
            apex_corners = []
            pt_sup, pt_inf = None, None
            if pt_sup is not None:
                pt_apex, pt_apex_corner = find_apex(pt_sup, pt_inf)
                apex_corners.append(pt_apex_corner)
            assert mt_sup is not None
            mt_apex, mt_apex_corner = find_apex(mt_sup, mt_inf)
            apex_corners.append(mt_apex_corner)
            assert lt_sup is not None
            lt_apex, lt_apex_corner = find_apex(lt_sup, lt_inf)
            apex_corners.append(lt_apex_corner)
        elif curve_algo == 'stored':  # Use stored value in json
            indices = lm_data['curve_indices']
            pt_sup, pt_inf = indices['PT']
            mt_sup, mt_inf = indices['MT']
            lt_sup, lt_inf = indices['TLL']
            apex_corners = []
            if pt_sup is not None:
                pt_apex, apex_corner = find_apex(pt_sup, pt_inf)
                apex_corners.append(apex_corner)
            if mt_inf is not None:
                if mt_inf >= len(vert_corners):
                    logger.info('Stored index exceeds possible value. %s, %s', lm_data['imagePath'], mt_inf)
                    mt_inf = len(vert_corners) - 1
                mt_apex, apex_corner = find_apex(mt_sup, mt_inf)
                apex_corners.append(apex_corner)
            if lt_inf is not None:
                if lt_inf >= len(vert_corners):
                    logger.info('Stored index exceeds possible value. %s, %s', lm_data['imagePath'], lt_inf)
                    lt_inf = len(vert_corners) - 1
            mt_apex, mt_apex_corner = find_apex(mt_sup, mt_inf)
            lt_apex, lt_apex_corner = find_apex(lt_sup, lt_inf)
            apex_corners = [mt_apex_corner, lt_apex_corner]
            if pt_sup is not None:
                pt_apex, pt_apex_corner = find_apex(pt_sup, pt_inf)
                apex_corners = [pt_apex_corner] + apex_corners
        else:
            raise NotImplemented

        measures['curve_indices'] = {'MT': (mt_sup, mt_inf), 'PT': (pt_sup, pt_inf), 'TLL': (lt_sup, lt_inf)}

        # cobb angles
        logger.debug('Cobb angles')
        drawings += [('MT', 'MT Cobb (p.49)'), ('TLL', 'TL/L Cobb (p.49)')]
        gid = 'MT'
        drawings.append((gid, 'MT Cobb (p.49)'))
        grp = dwg.add(
            dwg.g(
                id=gid,
                class_=gid,
                stroke=line_colors[gid],
                stroke_width=LINE_WIDTH,
                fill='none',
                visibility=d_visibility[gid],
            )
        )
        measures[gid] = dwg.draw_cobb(
            vert_sup_cnrs[mt_sup], vert_inf_cnrs[mt_inf], grp, title=gid, text_attrs=text_attrs
        )
        if lt_sup is not None and lt_inf is not None:
            gid = 'TLL'
        drawings.append((gid, 'TL/L Cobb (p.49)'))
        grp = dwg.add(
            dwg.g(
                id=gid,
                class_=gid,
                stroke=line_colors[gid],
                stroke_width=LINE_WIDTH,
                fill='none',
                visibility=d_visibility[gid],
            )
        )
        measures[gid] = dwg.draw_cobb(
            vert_sup_cnrs[lt_sup], vert_inf_cnrs[lt_inf], grp, title=gid, text_attrs=text_attrs
        )
        if pt_sup is not None:
            drawings += [('PT', 'PT Cobb (p.49)')]
            gid = 'PT'
            grp = dwg.add(
                dwg.g(
                    id=gid,
                    class_=gid,
                    stroke='red',
                    stroke_width=LINE_WIDTH,
                    fill='none',
                    visibility=d_visibility[gid],
                )
            )
            measures[gid] = dwg.draw_cobb(
                vert_sup_cnrs[pt_sup], vert_inf_cnrs[pt_inf], grp, title=gid, text_attrs=text_attrs
            )
        # Curve apex
        if apex_corners:
            gid = 'CurveApex'
            drawings.append((gid, 'Curve Apex(es)'))
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
            for vert_apex in apex_corners:
                grp.add(
                    dwg.polygon(
                        points=[vc.tolist() for vc in [vert_apex[0], vert_apex[1], vert_apex[3], vert_apex[2]]]
                    )
                )
            for v in vert_centroids:
                grp.add(dwg.circle(center=v, r=POINT_RADIUS, fill=color))

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
        grp.add(dwg.line((sacrum_tltr[0], sacrum_tltr[1])))
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
        grp = dwg.add(dwg.g(id=gid, class_=gid, stroke=color, visibility=d_visibility[gid]))

        sacrum_tr = sacrum_tltr[1]
        grp.add(dwg.polyline(sacrum_tr.tolist() + [sacrum_tr[0]] + [lerp(sacrum_tr[1], c7_centroid[1], 0.7)]))
        grp.add(dwg.polyline(c7_centroid.tolist() + [c7_centroid[0]] + [lerp(sacrum_tr[1], c7_centroid[1], 0.9)]))
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
            grp.add(dwg.line(vert_sup_cnrs[-1][0], vert_sup_cnrs[1]))
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
        for i, corners in enumerate(vert_corners_w_c7, 1):
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

    if measure_filename:
        with open(measure_filename, 'w') as f:
            json.dump(measures, f, ensure_ascii=False)

    if html_filename:
        env = Environment(loader=FileSystemLoader(template_directory, encoding='utf8'))
        template = env.get_template('template.html')
        t_cb = env.get_template('t_checkbox.html')

        cbs = []
        for name, label in drawings:
            cbs.append(t_cb.render(name=name, label=label, checked='' if name in hide else 'checked'))
        svg_tag = dwg.tostring()
        with open(html_filename, 'w', encoding='utf8') as f:
            f.write(
                template.render(svg=svg_tag, svg_container='document', checkboxes='\n'.join(cbs), title=html_title)
            )
    return measures


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
    parser.add_argument(
        '--algo',
        help='Algorithm to determine curves',
        choices=['max_diffs', 'fujimori', 'no-cross', 'stored'],
        default='max_diffs',
    )
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
        if args.parent_dir:
            image_dir = Path(args.parent_dir)
        else:
            image_dir = Path(args.input).parent
        with open(args.input) as f:
            lm_data = json.load(f)
        measure_json(
            lm_data,
            args.height,
            args.config,
            args.points,
            args.numbers,
            args.image_type,
            args.html,
            args.measure,
            args.template,
            args.title,
            args.hide,
            image_dir,
            args.slopes,
            args.algo,
            args.imagedb,
        )

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
        outdir = Path(args.output)
        if args.workers <= 0:
            import os

            args.workers = os.cpu_count()
            logger.info("Use %d cores", args.workers)
        import tqdm
        from joblib import Parallel, delayed

        def do(fn: Path):
            stem = re.sub(r"\.json$", "", fn.name)
            measure = outdir / (stem + '.json') if args.measure else None
            html = outdir / (stem + '.html') if args.html else None
            if args.parent_dir:
                image_dir = Path(args.parent_dir)
            else:
                image_dir = fn.parent
            with open(fn) as f:
                lm_data = json.load(f)
                lm_data['filename'] = fn.stem
            return measure_json(
                lm_data,
                args.height,
                args.config,
                args.points,
                args.numbers,
                args.image_type,
                html,
                measure,
                args.template,
                '{} - {}'.format(fn.stem, args.title),
                args.hide,
                image_dir,
                args.slopes,
                args.algo,
                args.imagedb,
            )

        def do_ndjson(fn: Path, lm_data):
            stem = re.sub(r"\.json$", "", Path(lm_data['filename']).name)
            measure = outdir / (stem + '.json') if args.measure else None
            html = outdir / (stem + '.html') if args.html else None
            if args.parent_dir:
                image_dir = Path(args.parent_dir)
            else:
                image_dir = fn.parent
            return measure_json(
                lm_data,
                args.height,
                args.config,
                args.points,
                args.numbers,
                args.image_type,
                html,
                measure,
                args.template,
                '{} - {}'.format(stem, args.title),
                args.hide,
                image_dir,
                args.slopes,
                args.algo,
                args.imagedb,
            )

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
                    ms['filename'] = re.sub(r"\.json$", "", fn.name)  # not using `fn.stem` to avoid stripping to much
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

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
