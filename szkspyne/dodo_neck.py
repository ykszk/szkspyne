import os
from pathlib import Path
import json
from logzero import logger
from szkspyne.utils.doit import Pipeline, concat
from szkspyne.utils.doit import make_phony, Pipeline
from szkspyne.settings import ASM_LABEL_NAMES
import szkspyne.utils.task as task_utils

RESULT_DIR = Path(os.environ['RESULT_DIR'])
os.environ['DATA_DIR'] = str(RESULT_DIR)

if not RESULT_DIR.exists():
    raise FileNotFoundError(f"RESULT_DIR {RESULT_DIR} does not exist. Please set the environment variable.")

ROOT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT_DIR / 'szkspyne'
SCRIPT_DIR = ROOT_DIR / 'script'

OUTPUT_MAIN_DIR = RESULT_DIR / 'postprocess'
if RESULT_DIR.exists():
    OUTPUT_MAIN_DIR.mkdir(exist_ok=True, parents=True)
FOLD_DIRS = list((RESULT_DIR / 'folds').glob('*'))

TGT_HM_KPS = 'heatmap_keypoints.ndjson'
HM_IMAGE_DB = OUTPUT_MAIN_DIR / f'heatmap_images.sqlite3'
PHONY_DB_TGT = 'hm2db.done'

TGT_PCA = 'pca_model.npz'


def task_build_pca():
    '''
    Create pca models
    '''
    script = SRC_DIR / 'asm/point_data.py'
    for fold_dir in FOLD_DIRS:
        # cat infile | python3 SRC_DIR/labelme_utils.py shape2lm | lmrs filter --rules SRC_DIR/conf/rules-neck.txt - | python3 SRC_DIR/labelme_utils.py lm2shape | python3 script outfile
        infile = fold_dir / 'train_val.ndjson'
        outfile = fold_dir / TGT_PCA

        pipeline = (
            Pipeline()
            .pipe(['cat', infile])
            .pipe(['python3', SRC_DIR / 'labelme_utils.py', 'shape2lm'])
            .pipe(['lmrs', 'filter', '--rules', SRC_DIR / 'conf/rules-neck.txt', '-'])
            .pipe(['python3', SRC_DIR / 'labelme_utils.py', 'lm2shape'])
            .pipe(['python3', script, outfile])
        )
        action = pipeline.runner(text=True)
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }


PCA_FIT_PARAMS = [
    '--sigma',
    '1',
    '0.1',
    '--max_iter',
    '400',
    '--l2_reg',
    '0.00001',
    '--device',
    'cpu',
    '--contr_rate',
    '0.9999',
    # '--jobs',
    # '8',
]


def task_fit_pca():
    '''
    Fit PCA models
    '''
    script = SRC_DIR / 'asm/pt_fit_model.py'
    for fold_dir in FOLD_DIRS:
        model_file = fold_dir / TGT_PCA
        outfile = fold_dir / 'auto/pca_keypoints.ndjson'
        mha_file = fold_dir / 'y_pred.mha'
        ndjson_file = fold_dir / 'auto/heatmap_keypoints.ndjson'
        skipped = fold_dir / 'auto/pca_skipped.txt'
        icp_weight = '0.00001'
        params = PCA_FIT_PARAMS + ['--skip_complete', '--skipped', skipped, '--icp_weight', icp_weight]

        action = ['python3', script, ndjson_file, mha_file, '--output', outfile] + params + ['--model', model_file]
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [model_file],
            'targets': [outfile],
        }


def task_merge_pca_hm():
    '''
    Merge PCA and heatmap keypoints
    '''
    for fold_dir in FOLD_DIRS:
        infile1 = fold_dir / 'auto/heatmap_keypoints.ndjson'
        infile2 = fold_dir / 'auto/pca_keypoints.ndjson'
        outfile = fold_dir / 'auto/pca_hm_keypoints.ndjson'
        # remove pca labels
        remove_labels = ASM_LABEL_NAMES + ["Lamina1", "C3_BL", "C3_BR"]
        # --label <name> to remove label
        label_args = []
        for label in remove_labels:
            label_args.append('--label')
            label_args.append(label)
        pipeline = Pipeline().pipe(['lmrs', 'remove', infile1] + label_args).pipe(['lmrs', 'merge', '-', infile2])
        action = pipeline.runner(text=True, output_file=outfile)
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile1, infile2],
            'targets': [outfile],
        }

        # filter
        #             ['lmrs', 'filter', '--rules', SRC_DIR / 'conf/rules-neck.txt', '-']
        infile = outfile
        outfile = fold_dir / 'auto/pca_hm_keypoints_filtered.ndjson'
        pipeline = Pipeline().pipe(['lmrs', 'filter', '--rules', SRC_DIR / 'conf/rules-neck.txt', infile])
        action = pipeline.runner(text=True, output_file=outfile)
        yield {
            'name': f'{fold_dir.name}-filter',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }


def task_heat2lm():
    '''
    Process heatmap to 1. create labelme files from heatmaps, 2. create heatmap image database
    '''
    for fold_dir in FOLD_DIRS:
        script = SRC_DIR / 'heatmap2labelme.py'
        infile = fold_dir / f'y_pred.mha'
        infile2 = fold_dir / f'test.ndjson'
        outdir = fold_dir / 'auto'
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / TGT_HM_KPS
        action = ['python', script, infile, infile2, outdir, '--save_ndjson', outfile]
        yield {
            'name': f'lm-{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

        script = SCRIPT_DIR / 'hm2db.py'
        infile = fold_dir / f'y_pred.mha'
        infile2 = outfile
        outfile = HM_IMAGE_DB
        phony = fold_dir / f'{PHONY_DB_TGT}'
        action = ['python', script, infile, infile2, outfile]
        yield {
            'name': f'db-{fold_dir.name}',
            'actions': [action, (make_phony, [phony])],
            'file_dep': [infile2],
            'targets': [phony],  # make phony target
        }


def task_ref_exp2lm():
    script = SCRIPT_DIR / 'exp2lmjson.py'
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / f'test.ndjson'
        outfile = fold_dir / 'lm_test.ndjson'
        action = ['python', script, infile]

        sort_corners = task_utils.CMD_SORT_VERTEBRAL_CORNERS
        sort_non_corners = sort_corners.copy() + ['--inv-label', '--by-x']

        sort_lamina = ['lmrs', 'sort', '-', '--labels', 'Lamina']

        pipeline = Pipeline().pipe(action).pipe(sort_corners).pipe(sort_non_corners).pipe(sort_lamina)
        action = pipeline.runner(text=True, output_file=outfile)
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }
    # concat
    infile = [fold_dir / 'lm_test.ndjson' for fold_dir in FOLD_DIRS]
    outfile = RESULT_DIR / 'postprocess/lm_test.ndjson'
    yield {
        'name': 'concat-lm-test',
        'actions': [(concat, [infile, outfile])],
        'file_dep': infile,
        'targets': [outfile],
    }

    # to native
    infile = outfile
    outfile = RESULT_DIR / 'postprocess/native_lm_test.ndjson'
    action = create_conv_action(infile, outfile)
    yield {
        'name': 'lm2native',
        'actions': [action],
        'file_dep': [infile],
        'targets': [outfile],
    }


def task_lm2native():
    # scolrs lm2native --input <input> --output <output>
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'auto/pca_hm_keypoints_filtered.ndjson'
        outfile = fold_dir / 'auto/native_pca_hm_keypoints.ndjson'
        action = create_conv_action(
            infile,
            outfile,
        )
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }


def create_conv_action(infile, outfile):
    # scolrs conv --ndjson --spacing <input> <output> --from Labelme --to LateralPoints
    action = ['scolrs', 'conv', '--ndjson', '--spacing', infile, outfile, '--from', 'Labelme', '--to', 'LateralPoints']
    return action


def task_ref_measure():
    # scolrs measure --ref --ndjson --output ref_measure.ndjson --input <input>
    for fold_dir in FOLD_DIRS:
        cmd = 'scolrs'
        infile = fold_dir / 'lm_test.ndjson'
        outfile = fold_dir / 'ref_measure.ndjson'
        action = [cmd, 'measure', '--labelme', infile, outfile, 'neck']
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

    # concat ref_measurements
    infile = [fold_dir / 'ref_measure.ndjson' for fold_dir in FOLD_DIRS]
    pipeline = Pipeline().pipe(['cat', *infile])
    outfile = RESULT_DIR / 'postprocess/measure_test_reference.ndjson'
    action = pipeline.runner(text=True, output_file=outfile)
    yield {
        'name': 'concat-ref-measurements',
        'actions': [action],
        'file_dep': infile,
        'targets': [outfile],
    }


SVG_IMAGE_SIZE = 800


def task_ref_draw():
    # scolrs svg-ndjson --resize 800x800 postprocess/lm_test.ndjson postprocess/reference/draw neck
    outdir = RESULT_DIR / 'postprocess/reference/draw'
    outdir.mkdir(parents=True, exist_ok=True)
    infile = RESULT_DIR / 'postprocess/native_lm_test.ndjson'
    config_dir = SRC_DIR / '../configs'
    line_colors = config_dir / 'default_line_colors_label.csv'
    label_colors = config_dir / 'default_label_colors.yaml'
    hides = ['Adi', 'Sacs', 'ModifiedRenawatIndex']
    action = [
        'scolrs',
        'svg-ndjson',
        '--html',
        '--label-colors',
        label_colors,
        '--line-colors',
        line_colors,
        '--resize',
        f'{SVG_IMAGE_SIZE}x{SVG_IMAGE_SIZE}',
        infile,
        outdir,
        'neck',
    ]
    for hide in hides:
        action += ['--hide', hide]

    phony = outdir / '../draw.done'
    phony_action = (make_phony, [phony])

    yield {
        'name': 'ref-draw',
        'actions': [action, phony_action],
        'file_dep': [infile],
        'targets': [phony],
    }


def task_ai_measure():
    # scolrs measure --ndjson --output ai_measure.ndjson --input <input>
    native_files = []
    for fold_dir in FOLD_DIRS:
        # measure
        cmd = 'scolrs'
        infile = fold_dir / 'auto/native_pca_hm_keypoints.ndjson'
        native_files.append(infile)
        outfile = fold_dir / 'auto/measurements.ndjson'
        action = [cmd, 'measure', infile, outfile, 'neck']
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

    # concat measurements
    infile = [fold_dir / 'auto/measurements.ndjson' for fold_dir in FOLD_DIRS]
    pipeline = Pipeline().pipe(['cat', *infile])
    outfile = RESULT_DIR / 'postprocess/measure_test_ai.ndjson'
    action = pipeline.runner(text=True, output_file=outfile)
    yield {
        'name': 'concat-measurements',
        'actions': [action],
        'file_dep': infile,
        'targets': [outfile],
    }

    # concat pca hm keypoints
    outfile = RESULT_DIR / 'auto/postprocess/native_pca_hm_keypoints.ndjson'
    pipeline = Pipeline().pipe(['cat', *native_files])
    action = pipeline.runner(text=True, output_file=outfile)
    yield {
        'name': 'concat-native',
        'actions': [action],
        'file_dep': native_files,
        'targets': [outfile],
    }
    # concat hm keypoints
    hm_files = [fold_dir / 'auto/heatmap_keypoints.ndjson' for fold_dir in FOLD_DIRS]
    hm_keypoints = RESULT_DIR / 'auto/postprocess/heatmap_keypoints.ndjson'
    outfile = hm_keypoints
    pipeline = Pipeline().pipe(['cat', *hm_files])
    action = pipeline.runner(text=True, output_file=outfile)
    yield {
        'name': 'concat-hm',
        'actions': [action],
        'file_dep': hm_files,
        'targets': [outfile],
    }

    # draw
    script = SRC_DIR / 'measure_and_draw.py'
    outdir = RESULT_DIR / 'auto/postprocess/draw'
    outdir.mkdir(parents=True, exist_ok=True)
    hide = ['Adi', 'Sacs', 'ModifiedRenawatIndex']
    infile = RESULT_DIR / 'auto/postprocess/native_pca_hm_keypoints.ndjson'
    action = [
        'python',
        script,
        '--resize',
        f'{SVG_IMAGE_SIZE}x{SVG_IMAGE_SIZE}',
        '--svg_size',
        str(SVG_IMAGE_SIZE),
        '--image_type',
        'neck',
        '--title',
        'AI Neck Measurements',
        '--hide',
        *hide,
    ]
    batch_args = [
        'batch',
        infile,
        outdir,
        '--html',
    ]
    subdir = 'auto'
    HOLD_OUT = False
    file_dep = [infile]
    extra_labels = []
    extra_points_files = []
    extra_labels.append('heatmap_points')
    extra_points_files.append(hm_keypoints)
    file_dep.append(extra_points_files[-1])
    # reference points
    extra_labels.append('reference_points')
    extra_points_files.append(RESULT_DIR / 'postprocess/lm_test.ndjson')
    file_dep.append(extra_points_files[-1])

    db_phonies = [fold_dir / f'{PHONY_DB_TGT}' for fold_dir in FOLD_DIRS]
    file_dep.extend(db_phonies)

    if subdir == 'auto':
        logger.debug(f'HM_IMAGE_DB: {HM_IMAGE_DB}')
        action += ['--imagedb', HM_IMAGE_DB]
    action += batch_args
    action += ['--extra_labels', *extra_labels]
    action += ['--extra_points', *extra_points_files]
    if HOLD_OUT:
        db_phony = OUTPUT_MAIN_DIR / f'holdout_{PHONY_DB_TGT}'
        file_dep.append(db_phony)
    draw_phony = RESULT_DIR / subdir / f'draw.done'
    yield {
        'name': f'draw-{subdir}',
        'actions': [action, (make_phony, [draw_phony])],
        'file_dep': file_dep,
        'targets': [draw_phony],
    }


def task_summary():
    script = ROOT_DIR / 'notebooks/analyze_neck_result.py'
    output = RESULT_DIR / 'summary.html'
    file_dep = [
        RESULT_DIR / 'postprocess/measure_test_reference.ndjson',
        RESULT_DIR / 'postprocess/measure_test_ai.ndjson',
    ]

    # deps of draw phony
    draw_phony = RESULT_DIR / 'auto/draw.done'
    file_dep.append(draw_phony)
    draw_phony = RESULT_DIR / 'postprocess/reference/draw.done'
    file_dep.append(draw_phony)

    title = f'{{"title":"neck summary"}}'
    c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
    c2 = f'jupyter nbconvert --stdin --to html --execute --no-input --output {output}'

    action = Pipeline().pipe(c1).pipe(c2).runner(text=True)
    return {
        'actions': [action],
        'file_dep': file_dep,
        'targets': [output],
    }
