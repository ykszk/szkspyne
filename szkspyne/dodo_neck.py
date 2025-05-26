import os
from pathlib import Path
import json
from logzero import logger
from szkspyne.utils.doit import Pipeline
from szkspyne.utils.doit import make_phony, Pipeline
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
            'name': f'{fold_dir.name}',
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

        pipeline = Pipeline().pipe(action).pipe(sort_corners).pipe(sort_non_corners)
        action = pipeline.runner(text=True, output_file=outfile)
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

def task_lm2native():
    # scolrs lm2native --input <input> --output <output>
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'lm_test.ndjson'
        outfile = fold_dir / 'native_lm_test.ndjson'
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

def task_ref_measure():
    # scolrs measure --ref --ndjson --output ref_measure.ndjson --input <input>
    for fold_dir in FOLD_DIRS:
        cmd = 'scolrs'
        infile = fold_dir / f'native_lm_test.ndjson'
        outfile = fold_dir / 'ref_measure.ndjson'
        action = [cmd, 'measure', infile, outfile, 'neck']
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

def task_ai_native():
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'auto' / TGT_HM_KPS
        outfile = fold_dir / 'auto/native_ai_test.ndjson'
        action = create_conv_action(infile, outfile)
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

def task_ai_measure():
    # scolrs measure --ndjson --output ai_measure.ndjson --input <input>
    for fold_dir in FOLD_DIRS:
        cmd = 'scolrs'
        infile = fold_dir / 'auto/native_ai_test.ndjson'
        outfile = fold_dir / 'ai_measure.ndjson'
        action = [cmd, 'measure', infile, outfile, 'neck']
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }


def task_summary():
    script = ROOT_DIR / 'notebooks/analyze_neck_result.py'
    output = RESULT_DIR / 'summary.html'
    file_dep = []

    title = f'{{"title":"neck summary"}}'
    c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
    c2 = f'jupyter nbconvert --stdin --to html --execute --no-input --output {output}'

    action = Pipeline().pipe(c1).pipe(c2).runner(text=True)
    return {
        'actions': [action],
        'file_dep': file_dep,
        'targets': [output],
    }
