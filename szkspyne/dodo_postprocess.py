import json
import os
import sys
from pathlib import Path
from typing import Optional

from logzero import logger

import szkspyne.utils.measure as measure_utils
import szkspyne.utils.task as task_utils
from szkspyne.utils.doit import Pipeline, concat, make_phony

log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logger.setLevel(log_level)

HOLD_OUT = os.environ.get('HOLD_OUT', False) == 'True'
logger.debug(f'HOLD_OUT: {HOLD_OUT}')
if HOLD_OUT:
    if 'MODEL_DIR' not in os.environ:
        logger.error('MODEL_DIR not set. Set MODEL_DIR to the directory containing the model files')
        sys.exit(1)
APPLY_CLAHE = os.environ.get('APPLY_CLAHE', False) == 'True'
logger.debug(f'APPLY_CLAHE: {APPLY_CLAHE}')

RESULT_DIR = Path(os.environ.get('RESULT_DIR', ''))
if RESULT_DIR == '':
    print('RESULT_DIR not set')
    sys.exit(1)
FOLD_DIRS = list((RESULT_DIR / 'folds').glob('*'))
ROOT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT_DIR / 'szkspyne'
SCRIPT_DIR = ROOT_DIR / 'script'
DOIT_CONFIG = {
    'verbosity': 2,
}

IMAGE_TYPES = ['ap_whole', 'lateral_whole']
IMAGE_TYPES = os.environ.get('IMAGE_TYPES', ','.join(IMAGE_TYPES)).split(
    ','
)  # Specify `IMAGE_TYPES=ap_whole` to process only ap_whole
logger.debug(f'IMAGE_TYPES: {IMAGE_TYPES}')

TGT_PCA = 'pca_ap_whole_L4.npz'

OUTPUT_MAIN_DIR = RESULT_DIR / 'postprocess'
if RESULT_DIR.exists():
    OUTPUT_MAIN_DIR.mkdir(exist_ok=True, parents=True)

RENDER_TRAIN = os.environ.get('RENDER_TRAIN', False) == 'True'


def task_split():
    '''
    Split y_pred.mha into subsets
    '''
    script = SCRIPT_DIR / 'split_exp_result.py'
    for fold_dir in FOLD_DIRS:
        infile1 = fold_dir / 'y_pred.mha'
        infile2 = fold_dir / 'test.ndjson'
        action = ['python', script, infile1, infile2, fold_dir]
        targets = [fold_dir / f'y_pred_{image_type}.mha' for image_type in IMAGE_TYPES]
        targets += [fold_dir / f'test_{image_type}.ndjson' for image_type in IMAGE_TYPES]
        yield {
            'name': fold_dir.name,
            'actions': [action],
            'file_dep': [infile1, infile2],
            'targets': targets,
        }


def task_pca():
    '''
    Create pca models
    '''
    script = SRC_DIR / 'pca_spine.py'
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'train_val.ndjson'
        outfile = fold_dir / TGT_PCA
        action = ['python', script, infile, fold_dir, '--mirror', '--ext']
        yield {
            'name': fold_dir.name,
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }


TGT_HM_KPS = 'all_heatmap_keypoints.ndjson'
HM_IMAGE_DB = OUTPUT_MAIN_DIR / 'heatmap_images.sqlite3'
PHONY_DB_TGT = 'hm2db.done'


def _task_holdout_heat2lm():
    for image_type in IMAGE_TYPES:
        infile = RESULT_DIR / 'auto' / 'output' / image_type
        infile2 = infile
        script = SCRIPT_DIR / 'hm2db.py'
        outfile = HM_IMAGE_DB
        action = ['python', script, infile, infile2, outfile]
        phony = OUTPUT_MAIN_DIR / f'holdout_{image_type}_{PHONY_DB_TGT}'
        apply_model_phony = RESULT_DIR / 'auto/output' / f'holdout_{image_type}_apply_model.done'
        yield {
            'name': f'db-holdout-{image_type}',
            'actions': [action, (make_phony, [phony])],
            'file_dep': [apply_model_phony],
            'targets': [phony],  # make phony target
        }

        command = ['lmrs', 'ndjson', str(infile), '--parent', 'remove']

        outfile = f'{RESULT_DIR}/auto/keypoints/{image_type}/heatmap_keypoints.ndjson'
        yield {
            'name': f'ndjson-{image_type}',
            'actions': [
                ' '.join(command) + f' > {outfile}',
            ],
            'file_dep': [phony],
            'targets': [outfile],
        }


def task_heat2lm():
    '''
    Process heatmap to 1. create labelme files from heatmaps, 2. create heatmap image database
    '''
    if HOLD_OUT:
        yield from _task_holdout_heat2lm()
        return
    for fold_dir in FOLD_DIRS:
        for image_type in IMAGE_TYPES:
            script = SRC_DIR / 'heatmap2labelme.py'
            infile = fold_dir / f'y_pred_{image_type}.mha'
            infile2 = fold_dir / f'test_{image_type}.ndjson'
            outdir = fold_dir / 'auto' / image_type
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / TGT_HM_KPS
            action = ['python', script, infile, infile2, outdir, '--save_ndjson', outfile]
            yield {
                'name': f'lm-{fold_dir.name}-{image_type}',
                'actions': [action],
                'file_dep': [infile],
                'targets': [outfile],
            }

            script = SCRIPT_DIR / 'hm2db.py'
            infile = fold_dir / f'y_pred_{image_type}.mha'
            infile2 = outfile
            outfile = HM_IMAGE_DB
            phony = fold_dir / f'{image_type}_{PHONY_DB_TGT}'
            action = ['python', script, infile, infile2, outfile]
            yield {
                'name': f'db-{fold_dir.name}-{image_type}',
                'actions': [action, (make_phony, [phony])],
                'file_dep': [infile2],
                'targets': [phony],  # make phony target
            }


PCA_FIT_PARAMS = [
    '--sigma',
    '1',
    '--max_iter',
    '400',
    '--l2_reg',
    '0.001',
    '--device',
    'cpu',
    '--jobs',
    '4',
]


def task_holdout_fit():
    if not HOLD_OUT:
        return
    script = SRC_DIR / 'fit_spine_model.py'
    for image_type in IMAGE_TYPES:
        indir = RESULT_DIR / 'auto/output' / image_type
        outdir = RESULT_DIR / 'auto/pca' / image_type
        outdir.mkdir(parents=True, exist_ok=True)

        model_dir = os.environ.get('MODEL_DIR')
        assert model_dir is not None
        model_dir = Path(model_dir)
        models = [
            '-m',
            model_dir / f'pca_{image_type}_L4.npz',
            model_dir / f'pca_{image_type}_L5.npz',
            model_dir / f'pca_{image_type}_L6.npz',
        ]
        icp_weight = '0.01'
        params = PCA_FIT_PARAMS + ['--skipped', outdir / f'skipped_{image_type}.txt', '--icp_weight', icp_weight]
        action = ['python', script, indir, '--output', outdir] + params + models

        ndjson_filename = RESULT_DIR / 'auto/keypoints' / image_type / 'pca_keypoints.ndjson'
        command = ['lmrs', 'ndjson', '--parent', 'remove', str(outdir)]
        action_create_ndjson = ' '.join(command) + f' > {ndjson_filename}'

        apply_model_phony = RESULT_DIR / 'auto/output' / f'holdout_{image_type}_apply_model.done'
        yield {
            'name': f'holdout-fit-{image_type}',
            'actions': [action, action_create_ndjson],
            'file_dep': [apply_model_phony],
            'targets': [ndjson_filename],
        }


def task_fit():
    '''
    Fit PCA model
    '''
    script = SRC_DIR / 'fit_spine_model.py'
    for fold_dir in FOLD_DIRS:
        for image_type in IMAGE_TYPES:
            infile = fold_dir / f'y_pred_{image_type}.mha'
            infile2 = fold_dir / 'auto' / image_type / TGT_HM_KPS
            pca_model = fold_dir / TGT_PCA
            outdir = fold_dir / 'auto' / image_type
            outfile = outdir / 'pca_keypoints.ndjson'

            params = PCA_FIT_PARAMS + ['--skipped', outdir / 'skipped.txt']
            models = [
                '-m',
                fold_dir / f'pca_{image_type}_L4.npz',
                fold_dir / f'pca_{image_type}_L5.npz',
                fold_dir / f'pca_{image_type}_L6.npz',
            ]
            action = ['python', script, infile2, infile, '--output', outfile] + params + models
            yield {
                'name': f'{fold_dir.name}-{image_type}',
                'actions': [action],
                'file_dep': [infile, infile2, pca_model],
                'targets': [outfile],
            }


# FLAGS = RESULT_DIR / 'flags.ndjson'
FLAGS = {
    'ap_whole': RESULT_DIR / 'postprocess/flags_ap_whole.ndjson',
    'lateral_whole': RESULT_DIR / 'postprocess/flags_lateral_whole.ndjson',
}


def task_flags():
    '''
    create flags.ndjson
    '''
    # jq =
    jqs = [
        '{"content": {"flags": .content.flags, "group_id": .content.group_id}, "filename": .filename}',
        # '.filename |= split("/")[-1]',
        '.content.flags = (.content.flags - ["L4","L5","L6"])',
        '.content.flags |= reduce .[] as $flag ({}; .[$flag] = true)',
    ]
    if HOLD_OUT:
        infile = RESULT_DIR / 'holdout_group_id_dicom_manual_annotation.ndjson'
    else:
        infile = RESULT_DIR / 'manual_annotation.ndjson'
    for image_type in IMAGE_TYPES:
        start = f"grep {image_type} {infile}"
        jq_pipe = [f"jq -c '{j}'" for j in jqs]
        sed = "sed 's/術後/PostOP/g'"
        action = ' | '.join([start, *jq_pipe, sed])
        target = FLAGS[image_type]
        action += f' > {target}'
        yield {'name': f'flags-{image_type}', 'actions': [action], 'file_dep': [infile], 'targets': [target]}


def task_collect():
    '''
    Colelct results from splits
    '''
    if HOLD_OUT:
        return
    for image_type in IMAGE_TYPES:
        infiles = [fold / f'auto/{image_type}/pca_keypoints.ndjson' for fold in FOLD_DIRS]
        outdir = RESULT_DIR / 'auto/keypoints' / image_type
        outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / 'pca_keypoints.ndjson'
        yield {
            'name': f'cat-{image_type}',
            'actions': [
                (
                    concat,
                    [infiles, outfile],
                )
            ],
            'file_dep': infiles,
            'targets': [outfile],
            'clean': True,
        }

        # heatmap points
        infiles = [fold / f'auto/{image_type}/all_heatmap_keypoints.ndjson' for fold in FOLD_DIRS]
        outdir = RESULT_DIR / 'auto/keypoints' / image_type
        outfile = outdir / 'heatmap_keypoints.ndjson'
        yield {
            'name': f'cat-hm-{image_type}',
            'actions': [
                (
                    concat,
                    [infiles, outfile],
                )
            ],
            'file_dep': infiles,
            'targets': [outfile],
            'clean': True,
        }

    # heatmap image db
    infiles = []
    for image_type in IMAGE_TYPES:
        infiles += [fold / f'{image_type}_{PHONY_DB_TGT}' for fold in FOLD_DIRS]
    outfile = RESULT_DIR / PHONY_DB_TGT
    yield {
        'name': 'db',
        'actions': [(make_phony, [outfile])],
        'file_dep': infiles,
        'targets': [outfile],
        'clean': True,
    }


def task_modify_keypoints():
    '''
    Modify keypoints
    '''
    for image_type in IMAGE_TYPES:
        script = SCRIPT_DIR / 'modify_ai_annotation.py'
        outdir = RESULT_DIR / 'auto/keypoints' / image_type
        outdir.mkdir(exist_ok=True, parents=True)
        infile = outdir / 'pca_keypoints.ndjson'
        outfile = outdir / 'final_keypoints.ndjson'
        proc = ' '.join([str(e) for e in ['python', script, image_type, infile, '--output', '-']])
        jq = "jq -c 'del(.pca)'"
        flags = FLAGS[image_type]
        join = f"lmrs join --mode left --missing continue - {flags}"
        sort_points = 'lmrs sort - --labels TL,TR,BL,BR --inv-label --by-x'
        pipeline = [proc, jq, join, sort_points]
        # if image_type == 'lateral_whole':
        #     pipeline = [' '.join(['cat', str(infile)]), jq, join, sort_points]
        action = ' | '.join(pipeline) + f' > {outfile}'
        yield {
            'name': f'mod-{image_type}',
            'actions': [action],
            'file_dep': [infile, flags],
            'targets': [outfile],
            'clean': True,
        }


def _task_holdout_ref():
    infile = RESULT_DIR / 'holdout_group_id_dicom_manual_annotation.ndjson'
    for image_type in IMAGE_TYPES:
        outdir = RESULT_DIR / f'reference/keypoints/{image_type}'
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / 'final_keypoints_exp.ndjson'
        jq_command = f'select(.content.flags | index("{image_type}")!=null)'

        yield {
            'name': f'holdout-ref-jq-{image_type}',
            'actions': [f"jq -c '{jq_command}' {infile} > {outfile}"],
            'file_dep': [],
            'targets': [outfile],
        }


def task_ref_collect():
    if HOLD_OUT:
        yield from _task_holdout_ref()
        return

    for image_type in IMAGE_TYPES:
        file_dep = [fold_dir / f'test_{image_type}.ndjson' for fold_dir in FOLD_DIRS]
        outdir = RESULT_DIR / f'reference/keypoints/{image_type}'
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / 'final_keypoints_exp.ndjson'

        yield {
            'name': f'cat-test-{image_type}',
            'actions': [
                (
                    concat,
                    [file_dep, outfile],
                )
            ],
            'file_dep': file_dep,
            'targets': [outfile],
            # 'clean': True,
        }


def task_ref_exp2lm():
    script = SCRIPT_DIR / 'exp2lmjson.py'
    for image_type in IMAGE_TYPES:
        outdir = RESULT_DIR / f'reference/keypoints/{image_type}'
        outdir.mkdir(parents=True, exist_ok=True)
        infile = outdir / 'final_keypoints_exp.ndjson'
        outfile = outdir / 'final_keypoints_unsorted.ndjson'

        yield {
            'name': f'exp2lm-{image_type}',
            'actions': [['python', script, infile, outfile]],
            'file_dep': [infile],
            'targets': [outfile],
        }
        infile = outfile
        # sort_command = f'lmrs sort {infile} --labels TL,TR,BL,BR | lmrs sort - --labels TL,TR,BL,BR --inv-label --by-x'
        cat = ['cat', str(infile)]
        sort_corners = task_utils.CMD_SORT_VERTEBRAL_CORNERS
        sort_non_corners = sort_corners.copy() + ['--inv-label', '--by-x']
        outfile = outdir / 'final_keypoints.ndjson'
        action = Pipeline().pipe(cat).pipe(sort_corners).pipe(sort_non_corners).runner(text=True, output_file=outfile)
        yield {
            'name': f'sort-{image_type}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

        # filename field to relative path
        infile = outfile
        outfile = outdir / 'final_keypoints_relpath.ndjson'
        jq_command = '.filename |= split("/")[-1]'
        yield {
            'name': f'relpath-{image_type}',
            'actions': [f"jq -c '{jq_command}' {infile} > {outfile}"],
            'file_dep': [infile],
            'targets': [outfile],
        }


AP_HIDE = (
    'T1TiltAngle CoronalBalance ClavicleAngle ShoulderHeight PelvicObliquity SacralObliquity LegLengthDiscrepancy'
)
LATERAL_HIDE = 'PelvicIncidence PelvicTilt SacralSlope L5IncidenceAngle PelvicRadiusAngle LumbosacralAngle'
hide_dict = {'ap_whole': AP_HIDE, 'lateral_whole': LATERAL_HIDE}


MES_HIDE = 'CurveApex SpinalLine APT CoronalBalance T1Tilt ClavicleAngle ShoulderHeight Pelvis FemoralHead PelvicObliquity LegLength SacralObliquity Mid/LowerThoracicKyphosis ProximalThoracicKyphosis SacralSlope PelvicIncidence PelvicRadiusAngle L5IncidenceAngle LumbosacralAngle'
JQ_REMOVE_BEND = '. | select(.content.flags."除外" or .content.flags."左屈" or .content.flags."右屈" or .content.flags."先天性側彎"|not)'
CURVE_ALGO = 'apex'

def task_curve():
    '''
    Determine curves
    '''
    for image_type in ['ap_whole']:
        for subdir in ['auto', 'reference']:
            file_deps = []
            infile = RESULT_DIR / subdir / 'keypoints' / image_type / 'final_keypoints.ndjson'
            file_deps.append(infile)
            curve_file = RESULT_DIR / subdir / f'curve_indices_{image_type}.ndjson'
            c1 = f"jq -c '{JQ_REMOVE_BEND}' {infile}"
            script = SCRIPT_DIR / 'sort_points.py'
            c2 = f'python {script} -'
            c3 = f'scolrs curve --algorithm {CURVE_ALGO} -'
            # c3 = f'scolrs curve -'
            script = SRC_DIR / 'curve.py'
            # c4 = f'python {script}'
            action1 = ' | '.join([c1, c2, c3]) + f' > {curve_file}'

            # join curves with points
            infile = RESULT_DIR / subdir / 'keypoints' / image_type / 'no-bend_final_keypoints_native.ndjson'
            file_deps.append(infile)
            curve_and_points_file = RESULT_DIR / subdir / 'curve_and_points.ndjson'
            action2 = f'lmrs join {infile} {curve_file} > {curve_and_points_file}'

            yield {
                'name': f'{image_type}-{subdir}',
                'actions': [action1, action2],
                'file_dep': [infile],
                'targets': [curve_file, curve_and_points_file],
                'clean': True,
            }


def task_closest_endplate():
    '''
    Find closest endplate
    '''
    script = SCRIPT_DIR / 'find_endplate.py'
    file_dep = [RESULT_DIR / 'reference/curve_and_points.ndjson']
    file_dep.append(RESULT_DIR / 'reference/keypoints/ap_whole/final_keypoints_relpath.ndjson')
    outdir = RESULT_DIR / 'closest'
    outfile = outdir / 'no-bend_final_keypoints_native.ndjson'
    action1 = ['python', script, RESULT_DIR]
    # draw
    draw_dir = outdir / 'ap_whole'
    draw_dir.mkdir(parents=True, exist_ok=True)
    action2 = [
        'scolrs',
        'svg-ndjson',
        outfile,
        draw_dir,
        '--resize',
        f'{SVG_IMAGE_SIZE}x{SVG_IMAGE_SIZE}',
        '--size',
        '1200x1200',
        '--html',
        '--label-colors',
        measure_utils.default_label_colors(),
        '--line-colors',
        measure_utils.default_line_colors(),
        'coronal',
        '--hide-group',
        'non-cobb',
    ]
    # measure
    output_measure = outdir / 'no-bend_measures_ap_whole.ndjson'
    action3 = ['scolrs', 'measure', outfile, output_measure, 'coronal', '--algo', CURVE_ALGO]
    # convert ndjson to csv
    script = SRC_DIR / 'utils/scolrs.py'
    action3_1 = ['python', script, output_measure, output_measure.with_suffix('.csv')]

    # compare measures
    script = SCRIPT_DIR / 'compare_measures.py'
    infile1 = RESULT_DIR / 'reference/no-bend_measures_ap_whole.ndjson'
    infile2 = output_measure
    outfile_ndjson = outdir / 'no-bend_measures_ap_whole_diff.ndjson'
    outfile_csv = outdir / 'no-bend_measures_ap_whole_diff.csv'
    action4 = ['python', script, infile1, infile2, outfile_ndjson, '--dataframe', outfile_csv, '--basename']
    yield {
        'name': 'find-endplate',
        'actions': [action1, action2, action3, action3_1, action4],
        # 'actions': [action1, action3, action4],
        'file_dep': file_dep,
        'targets': [outfile, outfile_ndjson, outfile_csv, output_measure, output_measure.with_suffix('.csv')],
    }


def task_summary_closest():
    '''
    Summarize closest endplate
    '''
    script = ROOT_DIR / 'notebooks/analyze_results.py'
    outdir = RESULT_DIR / 'closest'
    file_dep = [outdir / 'no-bend_measures_ap_whole_diff.csv']
    image_type = 'ap_whole'
    for post_op in [None, False, True]:
        c2 = f'AGAINST_CLOSEST=true RESULT_DIR={RESULT_DIR} IMAGE_TYPE={image_type} jupyter nbconvert --stdin --to html --execute --no-input'
        if post_op is not None:
            if post_op:
                outfile = outdir / f'summary_{image_type}_PostOp.html'
                c2 = 'POST_OP=True ' + c2 + ' --output ' + str(outfile)
                title = f'{{"title":"summary {image_type} PostOp"}}'
            else:
                outfile = outdir / f'summary_{image_type}_PreOp.html'
                c2 = 'POST_OP=False ' + c2 + ' --output ' + str(outfile)
                title = f'{{"title":"summary {image_type} PreOp"}}'
        else:
            outfile = outdir / f'summary_{image_type}.html'
            c2 = c2 + ' --output ' + str(outfile)
            title = f'{{"title":"summary {image_type}"}}'
        c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
        action = ' | '.join([c1, c2])
        yield {
            'name': f'summary-{image_type}-closest-{post_op}',
            'actions': [action],
            'file_dep': file_dep,
            'targets': [outfile],
            'clean': True,
        }


def remove_bend(infile, outfile):
    import subprocess

    with open(outfile, 'w') as wfd:
        subprocess.run(['jq', '-c', JQ_REMOVE_BEND, infile], stdout=wfd)


SVG_IMAGE_SIZE = 1600


def convert_to_native(infile, outfile, image_type, name):
    '''
    Yield task to convert labels to native format
    '''
    from_format = 'Labelme'
    to_format = 'ScoliosisCoronal' if image_type == 'ap_whole' else 'ScoliosisSagittal'
    yield {
        'name': name,
        'actions': [
            [
                'scolrs',
                'conv',
                infile,
                outfile,
                '--from',
                from_format,
                '--to',
                to_format,
                '--ndjson',
                '--spacing',
            ]
        ],
        'file_dep': [infile],
        'targets': [outfile],
    }


def task_native():
    '''
    Convert labels to native format
    '''
    db_phony = RESULT_DIR / PHONY_DB_TGT
    for image_type in IMAGE_TYPES:
        subdirs = ['reference', 'auto']
        for subdir in subdirs:
            infile = RESULT_DIR / subdir / 'keypoints' / image_type / 'final_keypoints.ndjson'
            outfile = infile.with_name('no-bend_final_keypoints.ndjson')
            file_dep = [infile, db_phony]
            if HOLD_OUT:
                file_dep = [infile]

            # remove bending
            yield {
                'name': f'remove-bend-{image_type}-{subdir}',
                'actions': [
                    (remove_bend, [infile, outfile]),
                ],
                'file_dep': file_dep,
                'targets': [outfile],
            }

            # convert from labelme to native
            infile = outfile
            outfile = infile.with_name('no-bend_final_keypoints_native.ndjson')

            yield from convert_to_native(infile, outfile, image_type, f'native-{image_type}-{subdir}')

            # measure
            infile = outfile
            outfile = RESULT_DIR / subdir / f'no-bend_measures_{image_type}.ndjson'
            if image_type == 'ap_whole':
                command = ["coronal", "--algo", CURVE_ALGO]
            else:
                command = ["sagittal"]
            yield {
                'name': f'measure-{image_type}-{subdir}',
                'actions': [['scolrs', 'measure', infile, outfile, *command]],
                'file_dep': [infile],
                'targets': [outfile],
            }

            # convert ndjson to csv
            script = SRC_DIR / 'utils/scolrs.py'
            yield {
                'name': f'ndjson2csv-{image_type}-{subdir}',
                'actions': [
                    ['python', script, outfile, outfile.with_suffix('.csv')],
                ],
                'file_dep': [outfile],
                'targets': [outfile.with_suffix('.csv')],
            }

            # draw
            script = SRC_DIR / 'measure_and_draw.py'
            outdir = RESULT_DIR / subdir / image_type
            outdir.mkdir(parents=True, exist_ok=True)
            action = [
                'python',
                script,
                '--resize',
                f'{SVG_IMAGE_SIZE}x{SVG_IMAGE_SIZE}',
                '--svg_size',
                '1200',
                '--image_type',
                image_type.split('_')[0],
                '--title',
                f'{subdir} {image_type}',
            ]
            batch_args = [
                'batch',
                infile,
                outdir,
                '--html',
                '--workers',
                '8',
            ]
            file_dep = [infile]
            extra_labels = ['all_points']
            extra_points_files = [infile.with_name('no-bend_final_keypoints.ndjson')]
            if subdir == 'auto':
                extra_labels.append('heatmap_points')
                extra_points_files.append(RESULT_DIR / subdir / 'keypoints' / image_type / 'heatmap_keypoints.ndjson')
                file_dep.append(extra_points_files[-1])
                # reference points
                extra_labels.append('reference_points')
                extra_points_files.append(
                    RESULT_DIR / 'reference' / 'keypoints' / image_type / 'final_keypoints_relpath.ndjson'
                )
                file_dep.append(extra_points_files[-1])

            if subdir == 'auto':
                logger.debug(f'HM_IMAGE_DB: {HM_IMAGE_DB}')
                action += ['--imagedb', HM_IMAGE_DB]
            action += batch_args
            action += ['--extra_labels', *extra_labels]
            action += ['--extra_points', *extra_points_files]
            if HOLD_OUT:
                db_phony = OUTPUT_MAIN_DIR / f'holdout_{image_type}_{PHONY_DB_TGT}'
                file_dep.append(db_phony)
            draw_phony = RESULT_DIR / subdir / f'draw_{image_type}.done'
            yield {
                'name': f'draw-{image_type}-{subdir}',
                'actions': [action, (make_phony, [draw_phony])],
                'file_dep': file_dep,
                'targets': [draw_phony],
            }

        scirpt = SCRIPT_DIR / 'compare_measures.py'
        infile1 = RESULT_DIR / 'reference' / f'no-bend_measures_{image_type}.ndjson'
        infile2 = RESULT_DIR / 'auto' / f'no-bend_measures_{image_type}.ndjson'
        outfile = OUTPUT_MAIN_DIR / f'no-bend_measures_{image_type}_diff.ndjson'
        outifle2 = OUTPUT_MAIN_DIR / f'no-bend_measures_{image_type}_diff.csv'
        yield {
            'name': f'diff-{image_type}',
            'actions': [['python', scirpt, infile1, infile2, outfile, '--dataframe', outifle2, '--basename']],
            'file_dep': [infile1, infile2],
            'targets': [outfile, outifle2],
        }


def task_apply_model():
    if not HOLD_OUT:
        return
    for image_type in IMAGE_TYPES:
        # create path file
        path_file = RESULT_DIR / f'auto/holdout_{image_type}_path.txt'

        jq_command1 = f'select(.content.flags | index("{image_type}")!=null)'
        jq_command2 = '.content.imagePath'

        ndjson_filename = 'holdout_group_id_dicom_manual_annotation.ndjson'
        action = f"jq -c '{jq_command1}' {RESULT_DIR}/{ndjson_filename} | jq -r '{jq_command2}' > {path_file}"
        yield {
            'name': f'extract-path-{image_type}',
            'actions': [action],
            'file_dep': [RESULT_DIR / ndjson_filename],
            'targets': [path_file],
        }

        # apply model
        script = SRC_DIR / 'apply_model.py'
        outdir = RESULT_DIR / 'auto/output' / image_type
        outdir.mkdir(parents=True, exist_ok=True)
        # search for the best model
        model_dir = os.environ.get('MODEL_DIR')
        assert model_dir is not None
        model_dir = Path(model_dir) / 'checkpoint'
        logger.debug(f'MODEL_DIR: {model_dir}')
        model_files = list(model_dir.glob('*.ckpt'))
        model_files.sort()
        model_file = model_files[-1]
        # read height
        with open(model_dir / '../args.json') as f:
            args = json.load(f)
            height = str(args['height'])
            # height = str(256 * 5)  # TODO: Implement two step model?
        auto_crop_multiplier = '1.0'
        action = [
            'python',
            script,
            'batch',
            path_file,
            '--ckpt',
            model_file,
            '--height',
            height,
            '--auto_crop_multiplier',
            auto_crop_multiplier,
            '--labelme',
            '--output',
            outdir,
            '--keep_image_path',
        ]
        if APPLY_CLAHE:
            action.append('--clahe')

        phony = RESULT_DIR / 'auto/output' / f'holdout_{image_type}_apply_model.done'

        yield {
            'name': f'apply-model-{image_type}',
            'actions': [action, (make_phony, [phony])],
            'file_dep': [path_file, model_file],
            'targets': [phony],
        }


def parse_optional_bool(str_bool: str) -> Optional[bool]:
    if str_bool == 'None':
        return None
    elif str_bool == 'True':
        return True
    elif str_bool == 'False':
        return False
    else:
        raise ValueError(f'Invalid boolean string: {str_bool}')


SUMMARY_POST_OPS = ['None', 'False', 'True']
SUMMARY_POST_OPS = os.environ.get('SUMMARY_POST_OPS', ','.join(SUMMARY_POST_OPS)).split(',')
SUMMARY_POST_OPS = [parse_optional_bool(v) for v in SUMMARY_POST_OPS]


def task_summary():
    ''' '
    Create summary
    '''
    file_dep = []
    script = ROOT_DIR / 'notebooks/analyze_results.py'
    for image_type in IMAGE_TYPES:
        for subdir in ['auto', 'reference']:
            dep = RESULT_DIR / subdir / f'no-bend_measures_{image_type}.csv'
            file_dep.append(dep)
            dep = RESULT_DIR / 'postprocess' / f'no-bend_measures_{image_type}_diff.csv'
            file_dep.append(dep)
            draw_phony = RESULT_DIR / subdir / f'draw_{image_type}.done'
            file_dep.append(draw_phony)

    for image_type in IMAGE_TYPES:
        for post_op in SUMMARY_POST_OPS:
            c2 = f'RESULT_DIR={RESULT_DIR} IMAGE_TYPE={image_type} jupyter nbconvert --stdin --to html --execute --no-input'
            if post_op is not None:
                if post_op:
                    outfile = OUTPUT_MAIN_DIR / f'summary_{image_type}_PostOp.html'
                    c2 = 'POST_OP=True ' + c2 + ' --output ' + str(outfile)
                    title = f'{{"title":"summary {image_type} PostOp"}}'
                else:
                    outfile = OUTPUT_MAIN_DIR / f'summary_{image_type}_PreOp.html'
                    c2 = 'POST_OP=False ' + c2 + ' --output ' + str(outfile)
                    title = f'{{"title":"summary {image_type} PreOp"}}'
            else:
                outfile = OUTPUT_MAIN_DIR / f'summary_{image_type}.html'
                c2 = c2 + ' --output ' + str(outfile)
                title = f'{{"title":"summary {image_type}"}}'
            c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
            action = ' | '.join([c1, c2])
            yield {
                'name': f'summary-{image_type}-{post_op}',
                'actions': [action],
                'file_dep': file_dep,
                'targets': [outfile],
                'clean': True,
            }


def _train_summary(source_file: Path, output_dir: Path):
    script = SCRIPT_DIR / 'exp2lmjson.py'
    infile = source_file
    first_lm_file = output_dir / (infile.stem + '_lm.ndjson')

    subname = str(source_file)

    yield {
        'name': f'exp2lm_{subname}',
        'actions': [['python', script, infile, first_lm_file]],
        'file_dep': [infile],
        'targets': [first_lm_file],
    }

    for image_type in IMAGE_TYPES:
        # split
        jq_command = f'select(.content.flags | has("{image_type}"))'  # `.content.flags` is dictionary of bool but just check the key existence is enough because the value should be always true
        infile = first_lm_file
        outfile = output_dir / (infile.stem + f'_{image_type}.ndjson')
        c1 = f"jq -c '{jq_command}' {infile}"
        c2 = f"jq -c '{JQ_REMOVE_BEND}'"
        sort = "lmrs sort -"
        action = ' | '.join([c1, c2, sort]) + f' > {outfile}'

        yield {
            'name': f'jq-{image_type}_{subname}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }
        # convert to native format
        infile = outfile
        # outfile = output_dir / f'manual_annotation_{image_type}_native.ndjson'
        outfile = output_dir / (source_file.stem + f'_{image_type}_native.ndjson')
        yield from convert_to_native(infile, outfile, image_type, f'to-native-{image_type}_{subname}')

        # measure
        infile = outfile
        outfile = output_dir / (source_file.stem + f'_{image_type}_measures.ndjson')
        if image_type == 'ap_whole':
            command = ["coronal", "--algo", CURVE_ALGO]
        else:
            command = ["sagittal"]
        yield {
            'name': f'measure-{image_type}_{subname}',
            'actions': [['scolrs', 'measure', infile, outfile, *command]],
            'file_dep': [infile],
            'targets': [outfile],
        }

        # convert ndjson to csv
        script = SRC_DIR / 'utils/scolrs.py'
        yield {
            'name': f'ndjson2csv-{image_type}_{subname}',
            'actions': [
                ['python', script, outfile, outfile.with_suffix('.csv')],
            ],
            'file_dep': [outfile],
            'targets': [outfile.with_suffix('.csv')],
        }

        # convert to html
        csv_filename = outfile.with_suffix('.csv')
        c2 = f'RESULT_DIR={RESULT_DIR} IMAGE_TYPE={image_type} CSV_FILENAME={csv_filename} FLAGS_FILENAME={source_file} jupyter nbconvert --stdin --to html --execute --no-input'
        script = ROOT_DIR / 'notebooks/analyze_train.py'
        c2 = c2 + ' --output ' + str(output_dir / f'{source_file.stem}_{image_type}.html')
        title = f'{{"title":"{source_file.stem} data {image_type}"}}'
        c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
        action = ' | '.join([c1, c2])
        yield {
            'name': f'summary-train-{image_type}_{subname}',
            'actions': [action],
            'file_dep': [outfile.with_suffix('.csv')],
            'targets': [],
            'clean': True,
        }

        # draw
        script = SRC_DIR / 'measure_and_draw.py'
        draw_dir = output_dir / image_type
        draw_dir.mkdir(parents=True, exist_ok=True)
        action = [
            'python',
            script,
            '--resize',
            f'{SVG_IMAGE_SIZE}x{SVG_IMAGE_SIZE}',
            '--svg_size',
            '1200',
            '--image_type',
            image_type.split('_')[0],
            '--title',
            f'{image_type}',
        ]
        batch_args = [
            'batch',
            infile,
            draw_dir,
            '--html',
            '--workers',
            '8',
        ]
        file_dep = [infile]
        action += batch_args
        yield {
            'name': f'draw-train-{image_type}-{subname}',
            'actions': [action],
            'file_dep': file_dep,
            'targets': [],
        }


def task_train_summary():
    '''
    Summary of training data
    '''
    output_dir = OUTPUT_MAIN_DIR / 'train'
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = RESULT_DIR / 'manual_annotation.ndjson'
    yield from _train_summary(
        source_file,
        output_dir,
    )

    for fold_dir in FOLD_DIRS:
        output_dir = OUTPUT_MAIN_DIR / f'train/fold{fold_dir.name}'
        output_dir.mkdir(parents=True, exist_ok=True)
        source_file = fold_dir / 'train_val.ndjson'
        yield from _train_summary(source_file, output_dir)
        source_file = fold_dir / 'test.ndjson'
        yield from _train_summary(source_file, output_dir)


def task_onnx():
    '''
    Convert labelme to onnx
    '''
    script = SRC_DIR / 'pretrained_models.py'
    model_dir = RESULT_DIR / 'folds/0/checkpoint'
    model_files = list(model_dir.glob('*.ckpt'))
    model_files.sort()
    if not model_files:
        return
    model_file = model_files[-1]
    outfile = model_dir / 'model.onnx'
    with open(model_dir / '../args.json') as f:
        args = json.load(f)
        height = str(args['height'])
        encoder = args['encoder_name']
        model_name = args['model_name']
    action = ['python', script, 'onnx', '--in_chs', '1', '--out_chs', '13', model_file, outfile, '--height', height]#, '--encoder_name', encoder, '--model_name', model_name]
    yield {
        'name': 'convert-onnx',
        'actions': [action],
        'file_dep': [model_file],
        'targets': [outfile],
    }
