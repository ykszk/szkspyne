import os
import sys
from pathlib import Path


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


def task_pca():
    '''
    Create pca models
    '''
    script = SRC_DIR / 'pca_spine.py'
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'train_val.ndjson'
        outfile = fold_dir / TGT_PCA
        action = ['python', script, infile, fold_dir, '--mirror']
        yield {
            'name': fold_dir.name,
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }


TGT_HM_KPS = 'all_heatmap_keypoints.ndjson'
HM_IMAGE_DB = RESULT_DIR / f'heatmap_images.sqlite3'


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
            action = ['python', script, infile, infile2, outfile]
            yield {
                'name': f'db-{fold_dir.name}-{image_type}',
                'actions': [action],
                'file_dep': [infile2],
                # 'targets': [outfile], # can't specify a common target
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
            outdir = fold_dir / 'auto' / image_type
            outfile = outdir / 'pca_keypoints.ndjson'
            params = [
                '--sigma',
                '1',
                '--max_iter',
                '400',
                '--l2_reg',
                '0.001',
                '--device',
                'cpu',
                '--skipped',
                outdir / 'skipped.txt',
                '--jobs',
                '4',
            ]
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
                'file_dep': [infile, infile2],
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


FLAGS = RESULT_DIR / 'flags.ndjson'


def task_flags():
    '''
    create flags.ndjson
    '''
    # jq =
    jqs = [
        '.filepath |= (split("/")[-1] |sub(".jpg";""))',
        '{"flags": .flags, "filename": .filepath}',
        '.flags = (.flags - ["L4","L5","L6"])',
        '.flags |= reduce .[] as $flag ({}; .[$flag] = true)',
    ]
    infile = RESULT_DIR / 'manual_annotation.ndjson'
    start = f'cat {infile}'  # TODO: remove linux dependency
    action = ' | '.join([start] + [f"jq -c '{j}'" for j in jqs])
    action += f' > {FLAGS}'
    return {'actions': [action], 'file_dep': [infile], 'targets': [FLAGS]}


def task_collect():
    '''
    Colelct results from splits
    '''
    for image_type in IMAGE_TYPES:
        infiles = list(RESULT_DIR.glob(f'folds/*/auto/{image_type}/pca_keypoints.ndjson'))
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
        script = SCRIPT_DIR / 'modify_ai_annotation.py'
        infile = outfile
        outfile = RESULT_DIR / 'auto/keypoints' / image_type / 'final_keypoints.ndjson'
        proc = ' '.join([str(e) for e in ['python', script, image_type, infile, '--output', '-']])
        jq = "jq -c 'del(.pca)'"
        join = f"lmrs join --mode left - {FLAGS}"
        action = ' | '.join([proc, jq, join]) + f' > {outfile}'
        yield {
            'name': f'mod-{image_type}',
            'actions': [action],
            'file_dep': [infile, FLAGS],
            'targets': [outfile],
            'clean': True,
        }


def task_ref():
    infile = RESULT_DIR / 'manual_annotation.ndjson'
    script = SCRIPT_DIR / 'exp2lmjson.py'
    for image_type in IMAGE_TYPES:
        c1 = f'grep {image_type} {infile}'
        outdir = RESULT_DIR / f'reference/keypoints/{image_type}'
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / 'final_keypoints.ndjson'
        c2 = f'python {script} - {outfile}'
        action = ' | '.join([c1, c2])
        yield {
            'name': image_type,
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
            'clean': True,
        }


MES_HIDE = 'CurveApex APT CoronalBalance T1Tilt ClavicleAngle ShoulderHeight Pelvis FemoralHead PelvicObliquity LegLength SacralObliquity Mid/LowerThoracicKyphosis ProximalThoracicKyphosis SacralSlope PelvicIncidence PelvicRadiusAngle L5IncidenceAngle LumbosacralAngle'
JQ_REMOVE_BEND = '. | select(.flags."除外" or .flags."左屈" or .flags."右屈" or .flags."先天性側彎"|not)'
ALGO = 'no-cross2'


def task_measure():
    '''
    Measure Cob angles
    '''
    for image_type in IMAGE_TYPES:
        for subdir in ['auto', 'reference']:
            infile = RESULT_DIR / subdir / 'keypoints' / image_type / 'final_keypoints.ndjson'
            curve_file = RESULT_DIR / subdir / f'curve_indices_{image_type}.ndjson'
            c1 = f"jq -c '{JQ_REMOVE_BEND}' {infile}"
            script = SRC_DIR / 'determine_curves.py'
            c2 = f'python {script} - {curve_file} --algo {ALGO} --jobs 4'
            action1 = ' | '.join([c1, c2])

            c2 = f'lmrs join --mode left - {curve_file}'
            script = SRC_DIR / 'measure_and_draw.py'
            parent_dir = SRC_DIR / '../../data/annotation/images/all'
            config = SRC_DIR / '../../data/annotation/config.yaml'
            outdir = RESULT_DIR / subdir / image_type
            ndjson = RESULT_DIR / subdir / f'measures_{image_type}.ndjson'
            imagedb = HM_IMAGE_DB
            c3 = f'python {script} --hide {MES_HIDE} --height 1200 --points --numbers --image_type {image_type.split("_")[0]} --parent_dir {parent_dir} --title "{subdir} {image_type}" --config {config} --imagedb {imagedb} batch - {outdir} --html --ndjson {ndjson} --workers 4'
            action2 = ' | '.join([c1, c2, c3])

            yield {
                'name': f'{image_type}-{subdir}',
                'actions': [action1, action2],
                'file_dep': [infile],
                'targets': [ndjson],
                'clean': True,
            }


def task_alt():
    '''
    Measure using alternative curves
    '''
    script = SCRIPT_DIR / 'find_best_curve_set.py'

    dep1 = RESULT_DIR / 'auto/curve_indices_ap_whole.ndjson'
    dep2 = RESULT_DIR / 'reference/measures_ap_whole.ndjson'
    out = RESULT_DIR / 'auto/best_indices.ndjson'
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
    out = RESULT_DIR / 'auto/closest_indices.ndjson'
    action = ['python', SCRIPT_DIR / 'find_endplate.py', RESULT_DIR]
    yield {
        'name': 'closest endplate',
        'actions': [action],
        'file_dep': [dep1, dep2, dep3, dep4],
        'targets': [out],
        'clean': True,
    }


def task_summary():
    ''' '
    Create summary
    '''
    script = ROOT_DIR / 'notebooks/exp_summary.py'
    title = '{"title":"exp summary"}'
    outfile = RESULT_DIR / 'summary.html'
    c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
    c2 = f'INPUT_DIR={RESULT_DIR} NO_VIS=true jupyter nbconvert --stdin --to html --execute --no-input --output {outfile}'
    action = ' | '.join([c1, c2])
    file_dep = []
    for image_type in IMAGE_TYPES:
        for subdir in ['auto', 'reference']:
            dep = RESULT_DIR / subdir / f'measures_{image_type}.ndjson'
            file_dep.append(dep)

    script = SCRIPT_DIR / 'measured_jsons2csv.py'
    pre = f'python3 {script} "{RESULT_DIR}"'
    return {
        'actions': [pre, action],
        'file_dep': file_dep,
        'targets': [outfile],
        'clean': True,
    }
