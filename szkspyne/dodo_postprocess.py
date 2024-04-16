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
    infile = RESULT_DIR / 'manual_annotation.ndjson'
    for image_type in IMAGE_TYPES:
        start = f"grep {image_type} {infile}"
        action = ' | '.join([start] + [f"jq -c '{j}'" for j in jqs])
        target = FLAGS[image_type]
        action += f' > {target}'
        yield {'name': f'flags-{image_type}', 'actions': [action], 'file_dep': [infile], 'targets': [target]}


def task_collect():
    '''
    Colelct results from splits
    '''
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
        script = SCRIPT_DIR / 'modify_ai_annotation.py'
        infile = outfile
        outfile = RESULT_DIR / 'auto/keypoints' / image_type / 'final_keypoints.ndjson'
        proc = ' '.join([str(e) for e in ['python', script, image_type, infile, '--output', '-']])
        jq = "jq -c 'del(.pca)'"
        flags = FLAGS[image_type]
        join = f"lmrs join --mode left - {flags}"
        action = ' | '.join([proc, jq, join]) + f' > {outfile}'
        yield {
            'name': f'mod-{image_type}',
            'actions': [action],
            'file_dep': [infile, flags],
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


MES_HIDE = 'CurveApex SpinalLine APT CoronalBalance T1Tilt ClavicleAngle ShoulderHeight Pelvis FemoralHead PelvicObliquity LegLength SacralObliquity Mid/LowerThoracicKyphosis ProximalThoracicKyphosis SacralSlope PelvicIncidence PelvicRadiusAngle L5IncidenceAngle LumbosacralAngle'
JQ_REMOVE_BEND = '. | select(.flags."除外" or .flags."左屈" or .flags."右屈" or .flags."先天性側彎"|not)'
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
            script = SRC_DIR / 'determine_curves.py'
            c2 = f'python {script} - {curve_file} --algo {ALGO} --jobs 4'
            action1 = ' | '.join([c1, c2])

            yield {
                'name': f'{image_type}-{subdir}',
                'actions': [action1],
                'file_dep': [infile],
                'targets': [curve_file],
                'clean': True,
            }


VARIANTS = ['best_indices', 'closest_indices']


def task_measure():
    '''
    Measure and draw into html files
    '''
    db_phony = RESULT_DIR / PHONY_DB_TGT
    for image_type in IMAGE_TYPES:
        for subdir in ['auto', 'reference']:
            for variant in [''] + VARIANTS:
                if variant != '' and image_type == 'lateral_whole':
                    continue
                if variant != '' and subdir == 'reference':
                    continue
                suffix = f'_{variant}' if variant != '' else ''
                infile = RESULT_DIR / subdir / 'keypoints' / image_type / f'final_keypoints.ndjson'
                file_dep = [infile, db_phony]

                # remove bending
                c1 = f"jq -c '{JQ_REMOVE_BEND}' {infile}"

                if image_type == 'ap_whole':
                    curve_file = RESULT_DIR / subdir / f'curve_indices_{image_type}{suffix}.ndjson'
                    c2 = f'lmrs join --mode left - {curve_file}'
                    file_dep.append(curve_file)
                else:
                    c2 = 'cat'  # do nothing
                    pass

                script = SRC_DIR / 'measure_and_draw.py'
                parent_dir = SRC_DIR / '../../data/annotation/images/all'
                config = SRC_DIR / '../../data/annotation/config.yaml'
                outdir = RESULT_DIR / subdir / f'{image_type}{suffix}'
                outdir.mkdir(parents=True, exist_ok=True)
                ndjson = RESULT_DIR / subdir / f'measures_{image_type}{suffix}.ndjson'
                imagedb = HM_IMAGE_DB
                c3 = f'python {script} --hide {MES_HIDE} --height 1200 --points --numbers --image_type {image_type.split("_")[0]} --parent_dir {parent_dir} --title "{subdir} {image_type}" --config {config} --imagedb {imagedb} batch - {outdir} --html --ndjson {ndjson} --workers 4'
                action = ' | '.join([c1, c2, c3])

                yield {
                    'name': f'{image_type}-{subdir}{suffix}',
                    'actions': [action],
                    'file_dep': file_dep,
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


def task_summary():
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
