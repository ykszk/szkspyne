import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tqdm
from joblib import Parallel, delayed
from logzero import logger
from skspatial.objects import LineSegment

from szkspyne.utils.measure import (
    CurveIndices,
    VertebraDiscIndex,
    Vertebrae,
    calc_angle,
    is_crossing,
)


def determine_curves(lm_data, curve_algo):
    vertebra_data = Vertebrae(lm_data)
    vert_corners = vertebra_data.vert_corners

    sacrum_center, vert_centroids = vertebra_data.sacrum_center, vertebra_data.vert_centroids
    vert_sup_cnrs, vert_inf_cnrs = (
        vertebra_data.vert_sup_cnrs,
        vertebra_data.vert_inf_cnrs,
    )
    curve_sets = []
    if curve_algo == 'max_diffs':
        best_set = None
        best_score = 0
        MT_MARGIN = 3
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
        curve_indices = CurveIndices.from_tuples((pt_sup, pt_inf), (mt_sup, mt_inf), (lt_sup, lt_inf))
        curve_sets.append({'name': 'Primary', 'indices': curve_indices.to_dict()})
    elif curve_algo == 'fujimori':
        coronal_translation = vert_centroids[:, 0] - sacrum_center[0]
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
        curve_indices = CurveIndices.from_tuples((pt_sup, pt_inf), (mt_sup, mt_inf), (lt_sup, lt_inf))
        curve_sets.append({'name': 'Primary', 'indices': curve_indices.to_dict()})
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
        lt_apex = vertebra_data.find_apex(lt_sup, lt_inf)
        if n_verts - lt_inf >= 2 and lt_apex < VertebraDiscIndex.T12:
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
        curve_indices = CurveIndices.from_tuples((pt_sup, pt_inf), (mt_sup, mt_inf), (lt_sup, lt_inf))
        curve_sets.append({'name': 'Primary', 'indices': curve_indices.to_dict()})
    elif curve_algo == 'no-cross2':
        # no-cross: max diff with constraint
        n_verts = len(vert_corners)

        largest_curve = vertebra_data.find_largest_angle()
        assert largest_curve is not None
        largest_sup, largest_inf, largest_angle, largest_apex = largest_curve

        sup, inf = largest_sup, largest_inf
        upwards: List[Vertebrae.Curve] = []
        downwards: List[Vertebrae.Curve] = []
        while sup >= 2:
            curve = vertebra_data.find_upward(sup)
            if curve is None:
                break
            next_sup, next_inf, next_angle, next_apex = curve
            assert next_sup < sup
            upwards.append(curve)
            sup = next_sup
        while inf <= n_verts - 2:
            curve = vertebra_data.find_downward(inf)
            if curve is None:
                break
            next_sup, next_inf, next_angle, next_apex = curve
            assert next_inf > inf
            downwards.append(curve)
            inf = next_inf
        all_curves = upwards + [largest_curve] + downwards
        all_angles = [c.angle for c in all_curves]
        idx_major = np.argmax(all_angles)
        major_apex = all_curves[idx_major].apex

        major_apexes = [major_apex]
        # add secondary curve-sets if primary apex is on the boundary
        if major_apex in [VertebraDiscIndex.DT4T5, VertebraDiscIndex.T5]:
            major_apexes.append(VertebraDiscIndex.T6)
        if major_apex in [VertebraDiscIndex.DT5T6, VertebraDiscIndex.T6]:
            major_apexes.append(VertebraDiscIndex.T5)

        if major_apex in [VertebraDiscIndex.T11, VertebraDiscIndex.DT11T12]:
            major_apexes.append(VertebraDiscIndex.T12)
        if major_apex in [VertebraDiscIndex.T12, VertebraDiscIndex.DT12L1]:
            major_apexes.append(VertebraDiscIndex.DT11T12)
        names = ['Primary', 'Secondary']
        for i, major_apex in enumerate(major_apexes):
            if major_apex <= VertebraDiscIndex.T5:  # major is pt
                apexes = {'PT': major_apex}
                pt_sup, pt_inf = largest_sup, largest_inf
                mt_sup, mt_inf = None, None
                lt_sup, lt_inf = None, None
                if len(downwards) >= 1:
                    mt_sup, mt_inf = downwards[0].sup, downwards[0].inf
                    apexes['MT'] = downwards[0].apex
                if len(downwards) >= 2:
                    lt_sup, lt_inf = downwards[1].sup, downwards[1].inf
                    apexes['LTT'] = downwards[1].apex
            elif major_apex <= VertebraDiscIndex.DT11T12:  # major is mt
                apexes = {'MT': major_apex}
                mt_sup, mt_inf = largest_sup, largest_inf
                if len(upwards) > 0:
                    pt_sup, pt_inf = upwards[0].sup, upwards[0].inf
                    apexes['PT'] = upwards[0].apex
                else:
                    pt_sup, pt_inf = None, None

                if len(downwards) > 0:
                    lt_sup, lt_inf = downwards[0].sup, downwards[0].inf
                    apexes['LTT'] = downwards[0].apex
                else:
                    lt_sup, lt_inf = None, None
            else:  # major is tll
                apexes = {'TLL': major_apex}
                lt_sup, lt_inf = largest_sup, largest_inf
                mt_sup, mt_inf = None, None
                pt_sup, pt_inf = None, None
                if len(upwards) >= 1:
                    mt_sup, mt_inf = upwards[0].sup, upwards[0].inf
                    apexes['MT'] = upwards[0].apex
                if len(upwards) >= 2:
                    pt_sup, pt_inf = upwards[1].sup, upwards[1].inf
                    apexes['PT'] = upwards[1].apex
            curve_indices = CurveIndices.from_tuples((pt_sup, pt_inf), (mt_sup, mt_inf), (lt_sup, lt_inf))
            angles = curve_indices.angles(vertebra_data)
            angles = {k: v for k, v in zip(['PT', 'MT', 'TLL'], angles)}
            curve_sets.append(
                {'name': names[i], 'indices': curve_indices.to_dict(), 'apices': apexes, 'angles': angles}
            )
    else:
        raise NotImplemented
    return curve_sets


def main():
    parser = argparse.ArgumentParser(description='Determine curves.')
    parser.add_argument('input', help='Input ndjson filename')
    parser.add_argument('output', help='Output ndjson filename')
    parser.add_argument('--algo', help='algorithm', default='no-cross')
    parser.add_argument('-j', '--jobs', help='jobs', default=0)

    args = parser.parse_args()
    if args.input == '-':
        ndjson = [json.loads(line) for line in sys.stdin]
    else:
        with open(args.input) as f:
            ndjson = [json.loads(line) for line in f]
    indir = Path(args.input).parent
    json_filenames = [indir / data['filename'] for data in ndjson]
    if args.jobs <= 0:
        args.jobs = os.cpu_count()
        logger.info("Use %d cores", args.jobs)
    all_curves = Parallel(n_jobs=args.jobs)(delayed(determine_curves)(data, args.algo) for data in tqdm.tqdm(ndjson))
    assert all_curves is not None
    with open(args.output, 'w') as f:
        for fn, curve_sets in zip(json_filenames, all_curves):
            line = {'filename': re.sub(r"\.json$", "", fn.name), 'curve_indices': curve_sets}
            f.write(json.dumps(line, ensure_ascii=False, separators=(',', ':')))
            f.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
