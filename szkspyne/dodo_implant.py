import os
from pathlib import Path
import json
from logzero import logger
from szkspyne.utils.doit import Pipeline

RESULT_DIR = Path(os.environ.get('RESULT_DIR', ''))
TRAINING_ONLY = os.environ.get('TRAINING_ONLY', 'false').lower() == 'true'

ROOT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT_DIR / 'szkspyne'
SCRIPT_DIR = ROOT_DIR / 'script'


def task_result_ndjson():
    if TRAINING_ONLY:
        return None
    indir = RESULT_DIR / 'ai/result'
    output = RESULT_DIR / 'ai/dtron2_results.ndjson'
    action = ['lmrs', 'ndjson', indir, '--parent', 'remove']
    runner = Pipeline().pipe(action).runner(text=True, output_file=output)

    return {'targets': [output], 'actions': [runner], 'file_dep': []}


def task_combine_dtron2_lm():
    if TRAINING_ONLY:
        return
    # lmrs join --mode inner  dtron2_results.ndjson lm_keypoints.ndjson > dtron2_lm_combined.ndjson
    dtron2_results = RESULT_DIR / 'ai/dtron2_results.ndjson'
    lm_keypoints = RESULT_DIR / 'ai/lm_keypoints.ndjson'
    output = RESULT_DIR / 'ai/dtron2_lm_combined.ndjson'

    action = ['lmrs', 'join', '--mode', 'inner', dtron2_results, lm_keypoints]
    runner = Pipeline().pipe(action).runner(text=True, output_file=output)
    return {
        'actions': [runner],
        'targets': [output],
        'file_dep': [dtron2_results],
    }


def task_catalog():
    # scolrs svg-ndjson dtron2_lm_combined - --resize 1600x1600 coronal-implant | scolrs catalog - catalog.html
    input_ndjson = RESULT_DIR / 'ai/dtron2_lm_combined.ndjson'
    catalog_html = RESULT_DIR / 'ai/catalog.html'

    IMG_SIZE = 1200

    if not TRAINING_ONLY:
        action = ['scolrs', 'svg-ndjson', input_ndjson, '-', '--resize', f'{IMG_SIZE}x{IMG_SIZE}', 'coronal-implant']
        action2 = ['scolrs', 'catalog', '-', catalog_html]

        catalog_action = Pipeline().pipe(action).pipe(action2).runner(text=True)

        yield {
            'name': 'ai',
            'actions': [catalog_action],
            'file_dep': [input_ndjson],
        }

    input_ndjson = RESULT_DIR / 'reference/lm_implant_keypoints.ndjson'
    catalog_html = RESULT_DIR / 'reference/catalog.html'
    action = ['scolrs', 'svg-ndjson', input_ndjson, '-', '--resize', f'{IMG_SIZE}x{IMG_SIZE}', 'coronal-implant']
    action2 = ['scolrs', 'catalog', '-', catalog_html]
    catalog_action = Pipeline().pipe(action).pipe(action2).runner(text=True)
    yield {
        'name': 'reference',
        'actions': [catalog_action],
        'file_dep': [input_ndjson],
        'targets': [catalog_html],
    }


def task_merge_refs():
    in1 = RESULT_DIR / 'reference/lm_keypoints.ndjson'
    in2 = RESULT_DIR / 'reference/lm_implant.ndjson'
    action = ['lmrs', 'merge', in1, in2, '--missing', 'exit']

    output = RESULT_DIR / 'reference/lm_implant_keypoints.ndjson'
    runner = Pipeline().pipe(action).runner(text=True, output_file=output)
    return {
        'actions': [runner],
        'file_dep': [in1, in2],
        'targets': [output],
    }


def task_count():
    # scolrs implant --count dtron2_lm_combined.ndjson > counts.ndjson

    in_ndjson = RESULT_DIR / 'reference/lm_implant_keypoints.ndjson'
    output = RESULT_DIR / 'reference/counts.ndjson'
    action = ['scolrs', 'implant', '--count', in_ndjson]
    runner = Pipeline().pipe(action).runner(text=True, output_file=output)
    yield {
        'name': 'reference',
        'actions': [runner],
        'file_dep': [in_ndjson],
        'targets': [output],
    }

    if TRAINING_ONLY:
        return
    in_ndjson = RESULT_DIR / 'ai/dtron2_lm_combined.ndjson'
    output = RESULT_DIR / 'ai/counts.ndjson'

    action = ['scolrs', 'implant', '--count', in_ndjson]
    runner = Pipeline().pipe(action).runner(text=True, output_file=output)
    yield {
        'name': 'ai',
        'actions': [runner],
        'file_dep': [in_ndjson],
        'targets': [output],
    }


def task_summary():
    script = ROOT_DIR / 'notebooks/analyze_implant_result.py'
    in1 = RESULT_DIR / 'ai/counts.ndjson'
    in2 = RESULT_DIR / 'reference/counts.ndjson'
    output = RESULT_DIR / 'summary.html'

    if TRAINING_ONLY:
        file_dep = [in2]
    else:
        file_dep = [in1, in2]

    title = f'{{"title":"implant summary"}}'
    c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
    # RESULT_DIR={RESULT_DIR}
    c2 = f'jupyter nbconvert --stdin --to html --execute --no-input --output {output}'

    action = Pipeline().pipe(c1).pipe(c2).runner(text=True)
    return {
        'actions': [action],
        'file_dep': file_dep,
        'targets': [output],
    }
