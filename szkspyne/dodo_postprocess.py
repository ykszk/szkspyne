import os
import sys
from pathlib import Path
from logzero import logger

log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logger.setLevel(log_level)

HOLD_OUT = bool(os.environ.get('HOLD_OUT', False))
logger.debug(f'HOLD_OUT: {HOLD_OUT}')

RESULT_DIR = Path(os.environ.get('RESULT_DIR', ''))
if RESULT_DIR == '':
    print('RESULT_DIR not set')
    sys.exit(1)
FOLD_DIRS = list((RESULT_DIR / 'folds').glob('*'))
ROOT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT_DIR / 'szkspyne'
SCRIPT_DIR = ROOT_DIR / 'script'

DOIT_CONFIG = {
    'default_tasks': ['_default'],
    'verbosity': 2,
}

IMAGE_TYPES = ['ap_whole', 'lateral_whole']

TGT_PCA = 'pca_ap_whole_L4.npz'

OUTPUT_MAIN_DIR = RESULT_DIR / 'postprocess'
if RESULT_DIR.exists():
    OUTPUT_MAIN_DIR.mkdir(exist_ok=True, parents=True)


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
HM_IMAGE_DB = RESULT_DIR / f'heatmap_images.sqlite3'
PHONY_DB_TGT = 'hm2db.done'


def make_phony(filename):
    with open(filename, 'w'):
        pass


def task_heat2lm():
    '''
    Process heatmap to 1. create labelme files from heatmaps, 2. create heatmap image database
    '''
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


#  python3 /poseidon/users/suzuki/projects/scoliosis/code/szkspyne/fit_spine_model.py auto/ap_whole/ --output pca/ap_whole/ -m ../exp20241113/folds/0/pca_ap_whole_L?.npz --sigma 1 --max_iter 400 --l2_reg 0.001 --skipped skipped_ap.txt --jobs 4
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

        model_dir = Path(os.environ.get('MODEL_DIR'))
        models = [
            '-m',
            model_dir / f'pca_{image_type}_L4.npz',
            model_dir / f'pca_{image_type}_L5.npz',
            model_dir / f'pca_{image_type}_L6.npz',
        ]
        params = PCA_FIT_PARAMS + ['--skipped', outdir / f'skipped_{image_type}.txt']
        action = ['python', script, indir, '--output', outdir] + params + models
        yield {
            'name': f'holdout-fit-{image_type}',
            'actions': [action],
            'file_dep': [],
            'targets': [],
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


def concat(files, output):
    import shutil

    if Path(output).exists():
        os.remove(output)
    with open(output, 'wb') as wfd:
        for f in files:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


# FLAGS = RESULT_DIR / 'flags.ndjson'
FLAGS = {'ap_whole': RESULT_DIR / 'flags_ap_whole.ndjson', 'lateral_whole': RESULT_DIR / 'flags_lateral_whole.ndjson'}


def task_flags():
    '''
    create flags.ndjson
    '''
    # jq =
    jqs = [
        '{"content": {"flags": .content.flags}, "filename": .filename}',
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
        action = ' | '.join([start] + [f"jq -c '{j}'" for j in jqs])
        target = FLAGS[image_type]
        action += f' > {target}'
        yield {'name': f'flags-{image_type}', 'actions': [action], 'file_dep': [infile], 'targets': [target]}


def _task_holdout_collect():
    for image_type in IMAGE_TYPES:
        indir = RESULT_DIR / 'auto/pca' / image_type
        outfile = RESULT_DIR / 'auto/keypoints' / image_type / 'pca_keypoints.ndjson'
        command = ['lmrs', 'ndjson', '--parent', 'remove', str(indir)]
        action = ' '.join(command) + f' > {outfile}'
        yield {
            'name': f'cat-ref-auto-{image_type}',
            'actions': [action],
            'file_dep': [],
            'targets': [outfile],
        }


def task_collect():
    '''
    Colelct results from splits
    '''
    if HOLD_OUT:
        yield from _task_holdout_collect()
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

    # heatmap image db
    infiles = []
    for image_type in IMAGE_TYPES:
        infiles += [fold / f'{image_type}_{PHONY_DB_TGT}' for fold in FOLD_DIRS]
    outfile = RESULT_DIR / PHONY_DB_TGT
    yield {
        'name': f'db',
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
        outfile = outdir / 'final_keypoints_.ndjson'

        yield {
            'name': f'exp2lm-{image_type}',
            'actions': [['python', script, infile, outfile]],
            'file_dep': [infile],
            'targets': [outfile],
        }
        infile = outfile
        sort_command = f'lmrs sort {infile} --labels TL,TR,BL,BR | lmrs sort - --labels TL,TR,BL,BR --inv-label --by-x'
        yield {
            'name': f'sort-{image_type}',
            'actions': [f'{sort_command} > {infile.with_name("final_keypoints.ndjson")}'],
            'file_dep': [infile],
            'targets': [infile.with_name('final_keypoints.ndjson')],
        }


AP_HIDE = (
    'T1TiltAngle CoronalBalance ClavicleAngle ShoulderHeight PelvicObliquity SacralObliquity LegLengthDiscrepancy'
)
LATERAL_HIDE = 'PelvicIncidence PelvicTilt SacralSlope L5IncidenceAngle PelvicRadiusAngle LumbosacralAngle'
hide_dict = {'ap_whole': AP_HIDE, 'lateral_whole': LATERAL_HIDE}


MES_HIDE = 'CurveApex SpinalLine APT CoronalBalance T1Tilt ClavicleAngle ShoulderHeight Pelvis FemoralHead PelvicObliquity LegLength SacralObliquity Mid/LowerThoracicKyphosis ProximalThoracicKyphosis SacralSlope PelvicIncidence PelvicRadiusAngle L5IncidenceAngle LumbosacralAngle'
JQ_REMOVE_BEND = '. | select(.content.flags."除外" or .content.flags."左屈" or .content.flags."右屈" or .content.flags."先天性側彎"|not)'
ALGO = 'no-cross2'


def task_curve():
    '''
    Determine curves
    '''
    for image_type in ['ap_whole']:
        for subdir in ['auto', 'reference']:
            infile = RESULT_DIR / subdir / 'keypoints' / image_type / 'final_keypoints.ndjson'
            curve_file = RESULT_DIR / subdir / f'curve_indices_{image_type}.ndjson'
            c1 = f"jq -c '{JQ_REMOVE_BEND}' {infile}"
            script = SCRIPT_DIR / 'sort_points.py'
            c2 = f'python {script} -'
            c3 = f'scolrs curve --all -'
            script = SRC_DIR / 'curve.py'
            c4 = f'python {script}'
            action1 = ' | '.join([c1, c2, c3, c4]) + f' > {curve_file}'

            yield {
                'name': f'{image_type}-{subdir}',
                'actions': [action1],
                'file_dep': [infile],
                'targets': [curve_file],
                'clean': True,
            }


def remove_bend(infile, outfile):
    import subprocess

    with open(outfile, 'w') as wfd:
        subprocess.run(['jq', '-c', JQ_REMOVE_BEND, infile], stdout=wfd)


SVG_IMAGE_SIZE = 1280


def task_native():
    '''
    Convert labels to native format
    '''
    db_phony = RESULT_DIR / PHONY_DB_TGT
    for image_type in IMAGE_TYPES:
        subdirs = ['reference', 'auto']
        for subdir in subdirs:
            infile = RESULT_DIR / subdir / 'keypoints' / image_type / f'final_keypoints.ndjson'
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
            from_format = 'Labelme'
            to_format = 'ScoliosisCoronal' if image_type == 'ap_whole' else 'ScoliosisSagittal'
            yield {
                'name': f'native-{image_type}-{subdir}',
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

            # measure
            infile = outfile
            outfile = RESULT_DIR / subdir / f'no-bend_measures_{image_type}.ndjson'
            command = "coronal" if image_type == "ap_whole" else "sagittal"
            yield {
                'name': f'measure-{image_type}-{subdir}',
                'actions': [['scolrs', 'measure', infile, outfile, command]],
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
            yield {
                'name': f'draw-{image_type}-{subdir}',
                'actions': [
                    [
                        'python',
                        script,
                        '--config',
                        SRC_DIR / '../../data/annotation/config.yaml',
                        '--resize',
                        f'{SVG_IMAGE_SIZE}x{SVG_IMAGE_SIZE}',
                        '--image_type',
                        image_type.split('_')[0],
                        '--title',
                        f'{subdir} {image_type}',
                        'batch',
                        infile,
                        outdir,
                        '--html',
                        '--workers',
                        '4',
                    ]
                ],
                'file_dep': [infile],
                'targets': [],
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


VARIANTS = ['best_indices', 'closest_indices']


def _task_measure():
    '''
    Measure and draw into html files
    '''
    db_phony = RESULT_DIR / PHONY_DB_TGT
    for image_type in IMAGE_TYPES:
        for subdir in ['auto', 'reference']:
            for variant in [''] + VARIANTS:
                pipeline = []
                if variant != '' and image_type == 'lateral_whole':
                    continue
                if variant != '' and subdir == 'reference':
                    continue
                suffix = f'_{variant}' if variant != '' else ''
                infile = RESULT_DIR / subdir / 'keypoints' / image_type / f'final_keypoints.ndjson'
                file_dep = [infile, db_phony]

                # remove bending
                c1 = f"jq -c '{JQ_REMOVE_BEND}' {infile}"
                pipeline.append(c1)

                if image_type == 'ap_whole':
                    curve_file = RESULT_DIR / subdir / f'curve_indices_{image_type}{suffix}.ndjson'
                    c2 = f'lmrs join --mode left - {curve_file}'
                    file_dep.append(curve_file)
                else:
                    c2 = 'cat'  # do nothing
                    pass
                pipeline.append(c2)

                # if subdir == 'reference':
                #     # extension of imagePath from dcm to jpg
                #     pipeline.append("jq -c '.content[\"imagePath\"] |= sub(\".dcm\"; \".jpg\")'")

                script = SRC_DIR / 'measure_and_draw.py'
                config = SRC_DIR / '../../data/annotation/config.yaml'
                outdir = RESULT_DIR / subdir / f'{image_type}{suffix}'
                outdir.mkdir(parents=True, exist_ok=True)
                imagedb = HM_IMAGE_DB
                c3 = f'python {script} --hide {MES_HIDE} --resize 1024x1024 --image_type {image_type.split("_")[0]} --title "{subdir} {image_type}" --config {config} --imagedb {imagedb} batch - {outdir} --html --workers 4'
                pipeline.append(c3)
                action = ' | '.join(pipeline)

                yield {
                    'name': f'draw-{image_type}-{subdir}{suffix}',
                    'actions': [action],
                    'file_dep': file_dep,
                    'targets': [],
                    'clean': True,
                }


def task_alt():
    '''
    Measure using alternative curves
    '''
    script = SCRIPT_DIR / 'find_best_curve_set.py'

    dep1 = RESULT_DIR / 'auto/curve_indices_ap_whole.ndjson'
    dep2 = RESULT_DIR / 'reference/measures_ap_whole.ndjson'
    out = RESULT_DIR / 'auto/curve_indices_ap_whole_best_indices.ndjson'
    action = ['python', script, RESULT_DIR]
    yield {
        'name': 'best curve',
        'actions': [action],
        'file_dep': [dep1, dep2],
        'targets': [out],
        'clean': True,
    }

    dep1 = RESULT_DIR / 'auto/keypoints/ap_whole/final_keypoints.ndjson'
    dep2 = RESULT_DIR / 'auto/measures_ap_whole.ndjson'
    dep3 = RESULT_DIR / 'reference/keypoints/ap_whole/final_keypoints.ndjson'
    dep4 = RESULT_DIR / 'reference/measures_ap_whole.ndjson'
    out = RESULT_DIR / 'auto/curve_indices_ap_whole_closest_indices.ndjson'
    action = ['python', SCRIPT_DIR / 'find_endplate.py', RESULT_DIR]
    yield {
        'name': 'closest endplate',
        'actions': [action],
        'file_dep': [dep1, dep2, dep3, dep4],
        'targets': [out],
        'clean': True,
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
        model_dir = Path(os.environ.get('MODEL_DIR')) / 'checkpoint'
        logger.debug(f'MODEL_DIR: {model_dir}')
        model_files = list(model_dir.glob('*.ckpt'))
        model_files.sort()
        model_file = model_files[-1]
        # python3 /poseidon/users/suzuki/projects/scoliosis/code/szkspyne/apply_model.py batch holdout_ap_whole_path.txt --ckpt ../exp20241113/folds/0/checkpoint/epoch=13-step=28210.ckpt --height 1024 --labelme --output auto/ap_whole/ --keep_image_path
        action = [
            'python',
            script,
            'batch',
            path_file,
            '--ckpt',
            model_file,
            '--height',
            '1024',
            '--labelme',
            '--output',
            outdir,
            '--keep_image_path',
            '--clahe',
        ]

        yield {
            'name': f'apply-model-{image_type}',
            'actions': [action],
            'file_dep': [path_file, model_file],
            'targets': [],
        }


def _task_summary():
    ''' '
    Create summary
    '''
    file_dep = []
    for image_type in IMAGE_TYPES:
        for subdir in ['auto', 'reference']:
            for variant in [None] + VARIANTS:
                if variant is not None and image_type == 'lateral_whole':
                    continue
                if variant is not None and subdir == 'reference':
                    continue
                suffix = f'_{variant}' if variant is not None else ''
                dep = RESULT_DIR / subdir / f'measures_{image_type}{suffix}.ndjson'
                file_dep.append(dep)
    targets = [RESULT_DIR / f'ap_whole.csv']
    for variant in VARIANTS:
        targets.append(RESULT_DIR / f'ap_whole_{variant}.csv')
    script = SCRIPT_DIR / 'measured_jsons2csv.py'
    pre = f'python3 {script} "{RESULT_DIR}"'

    yield {
        'name': 'pre',
        'actions': [pre],
        'file_dep': file_dep,
        'targets': targets,
        'clean': True,
    }

    script = ROOT_DIR / 'notebooks/exp_summary.py'
    for variant in [''] + VARIANTS:
        if variant == '':
            suffix = ''
            title = '{"title":"exp summary"}'
            name = 'normal'
        else:
            suffix = f'_{variant}'
            title = f'{{"title":"exp summary w {variant}"}}'
            name = variant
        file_dep = [RESULT_DIR / f'ap_whole{suffix}.csv']
        outfile = RESULT_DIR / f'summary{suffix}.html'
        c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
        c2 = f'INPUT_DIR={RESULT_DIR} NO_VIS=true DATA_SUFFIX={suffix} jupyter nbconvert --stdin --to html --execute --no-input --output {outfile}'
        action = ' | '.join([c1, c2])
        yield {
            'name': name,
            'actions': [action],
            'file_dep': file_dep,
            'targets': [outfile],
            'clean': True,
        }


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

    for image_type in IMAGE_TYPES:
        outfile = OUTPUT_MAIN_DIR / f'summary_{image_type}.html'
        title = f'{{"title":"summary {image_type}"}}'
        c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
        c2 = f'RESULT_DIR={RESULT_DIR} IMAGE_TYPE={image_type} jupyter nbconvert --stdin --to html --execute --no-input --output {outfile}'
        action = ' | '.join([c1, c2])
        yield {
            'name': f'summary-{image_type}-{subdir}',
            'actions': [action],
            'file_dep': file_dep,
            'targets': [outfile],
            'clean': True,
        }
