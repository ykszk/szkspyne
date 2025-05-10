import os
from pathlib import Path
import json
from logzero import logger
from szkspyne.utils.doit import Pipeline

RESULT_DIR = Path(os.environ.get('RESULT_DIR', ''))


ROOT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT_DIR / 'szkspyne'
SCRIPT_DIR = ROOT_DIR / 'script'


def task_result_ndjson():
    indir = RESULT_DIR / 'ai/result'
    output = RESULT_DIR / 'ai/dtron2_results.ndjson'
    action = ['lmrs', 'ndjson', indir, '--parent', 'remove']
    runner = Pipeline().pipe(action).runner(text=True, output_file=output)

    return {'targets': [output], 'actions': [runner], 'file_dep': []}


def save_file(filename):
    def save_file_action(content):
        with open(filename, 'w') as f:
            f.write(content)

    return save_file_action


def task_combine_dtron2_lm():
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
    dtron2_lm_combined = RESULT_DIR / 'ai/dtron2_lm_combined.ndjson'
    catalog_html = RESULT_DIR / 'ai/catalog.html'

    action = ['scolrs', 'svg-ndjson', dtron2_lm_combined, '-', '--resize', '1600x1600', 'coronal-implant']
    action2 = ['scolrs', 'catalog', '-', catalog_html]

    catalog_action = Pipeline().pipe(action).pipe(action2).runner(text=True)

    return {
        'actions': [catalog_action],
        'file_dep': [dtron2_lm_combined],
    }


def task_merge_refs():
    in1 = RESULT_DIR / 'reference/lm_implant.ndjson'
    in2 = RESULT_DIR / 'reference/lm_keypoints.ndjson'
    action = ['lmrs', 'merge', in1, in2, '--missing', 'continue']

    output = RESULT_DIR / 'reference/lm_implant_keypoints.ndjson'
    runner = Pipeline().pipe(action).runner(text=True, output_file=output)
    return {
        'actions': [runner],
        'file_dep': [in1, in2],
        'targets': [output],
    }


def task_count():
    # scolrs implant --count dtron2_lm_combined.ndjson > counts.ndjson
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


def task_summary():
    script = ROOT_DIR / 'notebooks/analyze_implant_result.py'
    in1 = RESULT_DIR / 'ai/counts.ndjson'
    in2 = RESULT_DIR / 'reference/counts.ndjson'
    output = RESULT_DIR / 'summary.html'

    title = f'{{"title":"implant summary"}}'
    c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
    # RESULT_DIR={RESULT_DIR}
    c2 = f'jupyter nbconvert --stdin --to html --execute --no-input --output {output}'

    action = Pipeline().pipe(c1).pipe(c2).runner(text=True)
    return {
        'actions': [action],
        'file_dep': [in1, in2],
        'targets': [output],
    }
