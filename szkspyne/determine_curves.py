import sys
import argparse
import json
import re
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from szkspyne.utils.measure import (
    Vertebrae,
    CurveIndices,
    calc_angle,
    is_crossing,
    find_apex,
)
import numpy as np
from skspatial.objects import Line, LineSegment
from logzero import logger
import tqdm
from joblib import Parallel, delayed


def determine_curves(lm_data):
    vertebra_data = Vertebrae(lm_data)
    vert_corners = vertebra_data.vert_corners

    sacrum_center, vert_centroids = vertebra_data.sacrum_center, vertebra_data.vert_centroids
    vert_sup_cnrs, vert_inf_cnrs = (
        vertebra_data.vert_sup_cnrs,
        vertebra_data.vert_inf_cnrs,
    )
    curve_algo = 'no-cross'
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
        lt_apex = find_apex(vert_corners, vert_centroids, lt_sup, lt_inf)
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
        curve_indices = CurveIndices.from_tuples((pt_sup, pt_inf), (mt_sup, mt_inf), (lt_sup, lt_inf))
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
        major_apex = find_apex(vert_corners, vert_centroids, major_sup, major_inf)
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
                lt_sup, lt_inf = None, None
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
                mt_sup, mt_inf = None, None
            else:
                mt_sup, mt_inf = sup_inf[np.argmax(scores)]

        pt_sup, pt_inf = None, None
        curve_indices = CurveIndices.from_tuples((pt_sup, pt_inf), (mt_sup, mt_inf), (lt_sup, lt_inf))
    else:
        raise NotImplemented
    return curve_indices.to_dict()


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
    all_measures = Parallel(n_jobs=args.jobs)(delayed(determine_curves)(data) for data in tqdm.tqdm(ndjson))
    assert all_measures is not None
    with open(args.output, 'w') as f:
        for fn, ms in zip(json_filenames, all_measures):
            line = {'filename': re.sub(r"\.json$", "", fn.name), 'curve_indices': ms}
            f.write(json.dumps(line, ensure_ascii=False, separators=(',', ':')))
            f.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
