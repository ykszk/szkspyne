import base64
import gzip
import json
import os
import shutil
from pathlib import Path

from bs4 import BeautifulSoup
from logzero import logger

import szkspyne.utils.task as task_utils
from szkspyne.utils.doit import Pipeline, concat, make_phony

RESULT_DIR = Path(os.environ['RESULT_DIR'])
os.environ['DATA_DIR'] = str(RESULT_DIR)
# os.environ['DS__HEATMAP__THRESH'] = '0.1' # default is 0.1
os.environ['DS__HEATMAP__BLUR_SIGMA'] = '3' # default is none

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

TGT_PCA = 'pca_model.npz'
TGT_PCA_JSON = 'pca_model.json'

EXTERNAL_TEST = os.environ.get('EXTERNAL_TEST', 'False').lower() in ('true', '1', 'yes')

NECK_STUDY_POSES = RESULT_DIR / "neck_study_poses.json"

# Rotate points slightly so that the vertebral corners are sorted by y coordinate. Some flexion images have caudal vertebra with lower y coordinate than cranial vertebra.
CMD_ROTATE = ['lmrs', 'mat', '-r=5', '-', '--ndjson']
CMD_ROTATE_INV = ['lmrs', 'mat', '-r=-5', '-', '--ndjson']

CMD_SORT_LAMINA = ['lmrs', 'sort', '-', '--labels', 'Lamina']
CMD_SORT_NON_VERTEBRAL_AND_LAMINA = task_utils.CMD_SORT_VERTEBRAL_CORNERS + ['--labels', 'Lamina', '--inv-label', '--by-x']

AI_SUBDIR = 'auto'
if EXTERNAL_TEST:
    # to differentiate file dependencies of external test from regular tasks
    AI_SUBDIR = 'ai'

if EXTERNAL_TEST:
    FOLDS_DIRS = [RESULT_DIR / 'folds/0']
    (FOLD_DIRS[0] / 'ai').mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / 'postprocess').mkdir(parents=True, exist_ok=True)


def extract_measurements_from_html(html_path) -> str:
    with open(html_path) as f:
        soup = BeautifulSoup(f, 'html.parser')
    mes_json = soup.find(id="embedded-measurements.json")
    assert mes_json is not None, f"No element with id 'embedded-measurements.json' found in {html_path}"

    json_str = gzip.decompress(base64.b64decode(mes_json.text.strip())).decode('utf-8')
    return json_str


def task_create_onnx():
    '''
    Convert PyTorch model to ONNX format for external test
    '''
    if EXTERNAL_TEST:
        return
    script = SRC_DIR / 'pretrained_models.py'
    for fold_dir in FOLD_DIRS:
        logger.info(f'Processing fold directory: {fold_dir}')
        model_dir = fold_dir / "checkpoint"
        assert model_dir is not None
        model_dir = Path(model_dir)
        model_files = list(model_dir.glob('*.ckpt'))
        model_files.sort()
        model_file = model_files[-1]
        output_file = fold_dir / 'model.onnx'
        action = ['python3', script, 'onnx', str(model_file), str(output_file)]
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [model_file],
            'targets': [output_file],
        }


def task_build_pca():
    '''
    Create pca models
    '''
    script = SRC_DIR / 'asm/point_data.py'
    for fold_dir in FOLD_DIRS:
        # cat infile | python3 SRC_DIR/labelme_utils.py shape2lm | lmrs filter --rules SRC_DIR/conf/rules-neck.txt - | python3 SRC_DIR/labelme_utils.py lm2shape | python3 script outfile
        infile = fold_dir / 'train_val.ndjson'
        outfile = fold_dir / TGT_PCA
        json_outfile = fold_dir / TGT_PCA_JSON

        pipeline = (
            Pipeline()
            .pipe(['cat', infile])
            .pipe(['python3', SRC_DIR / 'labelme_utils.py', 'shape2lm'])
            .pipe(['lmrs', 'filter', '--rules', SRC_DIR / 'conf/rules-neck.txt', '-'])
            .pipe(CMD_ROTATE)
            .pipe(task_utils.CMD_SORT_VERTEBRAL_CORNERS)
            .pipe(CMD_SORT_LAMINA)
            .pipe(CMD_ROTATE_INV)
            .pipe(CMD_SORT_NON_VERTEBRAL_AND_LAMINA)
            .pipe(['python3', SRC_DIR / 'labelme_utils.py', 'lm2shape'])
            .pipe(['python3', script, outfile, '--json', json_outfile])
        )
        action = pipeline.runner(text=True)
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile, json_outfile],
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


def sub_exp2lm(name, infile, outfile):
    '''
    Convert exp results to lm format for external test
    '''
    script = SCRIPT_DIR / 'exp2lmjson.py'
    if EXTERNAL_TEST:
        action = ['cat', infile]
    else:
        action = ['python', script, infile]

    sort_corners = task_utils.CMD_SORT_VERTEBRAL_CORNERS
    sort_non_corners = CMD_SORT_NON_VERTEBRAL_AND_LAMINA

    pipeline = Pipeline().pipe(action).pipe(CMD_ROTATE).pipe(sort_corners).pipe(CMD_SORT_LAMINA).pipe(CMD_ROTATE_INV).pipe(sort_non_corners)
    action = pipeline.runner(text=True, output_file=outfile)
    return {
        'name': name,
        'actions': [action],
        'file_dep': [infile],
        'targets': [outfile],
    }

def task_ref_exp2lm():
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'test.ndjson'
        outfile = fold_dir / 'lm_test.ndjson'
        yield sub_exp2lm(f'{fold_dir.name}', infile, outfile)

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

    # normalize vertebral shape
    infile = outfile
    outfile = RESULT_DIR / 'postprocess/normalized_native_lm_test.ndjson'
    if NECK_STUDY_POSES.exists():
        action = _normalize_vertebral_shape(infile, NECK_STUDY_POSES, outfile)
    else:
        logger.warning(f"NECK_STUDY_POSES {NECK_STUDY_POSES} does not exist. Skipping shape normalization.")
        action = ['cp', infile, outfile]
    yield {
        'name': 'normalize-shape',
        'actions': [action],
        'file_dep': [infile],
        'targets': [outfile],
    }


def create_conv_action(infile, outfile):
    # scolrs conv --ndjson --spacing <input> <output> --from Labelme --to LateralPoints
    action = ['scolrs', 'conv', '--ndjson', '--spacing', infile, outfile, '--from', 'Labelme', '--to', 'LateralPoints']
    return action


def task_ref_measure():
    infile = RESULT_DIR / 'postprocess/normalized_native_lm_test.ndjson'
    outfile = RESULT_DIR / 'postprocess/measure_test_reference.ndjson'
    action = ['scolrs', 'measure', infile, outfile, 'neck']
    return {
        'actions': [action],
        'file_dep': [infile],
        'targets': [outfile],
    }


SVG_IMAGE_SIZE = 800

HIDES = ['Adi', 'Sacs', 'RedlundJohnellMethod', 'RanawatMethod', 'ModifiedRanawatIndex', 'Tpr', 'EndPlateAngle', 'Eacsva', 'C2c7sva', 'MidPlaneAngle', 'InterVertebralAngle', 'AnteroposteriorVertebralTranslation']

def task_ref_draw():
    # scolrs svg-ndjson --resize 800x800 postprocess/lm_test.ndjson postprocess/reference/draw neck
    outdir = RESULT_DIR / 'postprocess/reference/draw'
    outdir.mkdir(parents=True, exist_ok=True)
    infile = RESULT_DIR / 'postprocess/normalized_native_lm_test.ndjson'
    config_dir = SRC_DIR / '../configs'
    line_colors = config_dir / 'default_line_colors_label.csv'
    label_colors = config_dir / 'default_label_colors.yaml'
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
    for hide in HIDES:
        action += ['--hide', hide]

    phony = (outdir / '../draw.done').absolute()
    phony_action = (make_phony, [phony])

    yield {
        'name': 'ref-draw',
        'actions': [action, phony_action],
        'file_dep': [infile],
        'targets': [phony],
    }


def task_ai_measure():
    # concat measurements
    infile = [str(fold_dir / AI_SUBDIR / 'measurements.ndjson') for fold_dir in FOLD_DIRS]
    pipeline = Pipeline().pipe(['cat', *infile])
    outfile = RESULT_DIR / 'postprocess/measure_test_ai.ndjson'
    action = pipeline.runner(text=True, output_file=outfile)
    yield {
        'name': 'concat-measurements',
        'actions': [action],
        'file_dep': infile,
        'targets': [outfile],
    }

def task_train_exp2lm():
    # exp to labelme format
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'train_val.ndjson'
        outfile = fold_dir / 'train_lm.json'
        yield sub_exp2lm(f'{fold_dir.name}', infile, outfile)

def task_train_lm2native():
    # train labelme to native format
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'train_lm.json'
        outfile = fold_dir / 'train_native.ndjson'
        action = create_conv_action(infile, outfile)
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

def _normalize_vertebral_shape(infile, poses, outfile):
    c1 = ['python', SRC_DIR / 'type/neck_study.py', poses, infile]
    c2 = ['scolrs', 'vertebra', 'normalize', "-"]
    # remove empty lines
    c3 = ['awk', 'NF']
    pipeline = Pipeline().pipe(c1).pipe(c2).pipe(c3)
    action = pipeline.runner(text=True, output_file=outfile)
    return action


def task_train_normalize_shape():
    # normalize vertebral shapes
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'train_native.ndjson'
        outfile = fold_dir / 'train_normalized_native.ndjson'
        if NECK_STUDY_POSES.exists():
            action = _normalize_vertebral_shape(infile, NECK_STUDY_POSES, outfile)
        else:
            logger.warning(f"NECK_STUDY_POSES {NECK_STUDY_POSES} does not exist. Skipping shape normalization.")
            action = ['cp', infile, outfile]
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

def task_train_measure():
    # measure train data
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'train_normalized_native.ndjson'
        outfile = fold_dir / 'measure_train.ndjson'
        action1 = ['scolrs', 'measure', infile, outfile, 'neck']
        # ndjson to csv
        script = SRC_DIR / 'utils/scolrs.py'
        action2 = ['python', script, outfile, outfile.with_suffix('.csv')]

        yield {
            'name': fold_dir.name,
            'actions': [action1, action2],
            'file_dep': [infile],
            'targets': [outfile],
        }
    
def task_train_rom():
    # calculate ROM of train data
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / 'measure_train.ndjson'
        outfile = fold_dir / 'rom_train.csv'
        script = SCRIPT_DIR / 'calculate_rom.py'
        action = ['python', script, NECK_STUDY_POSES, infile, outfile]
        yield {
            'name': fold_dir.name,
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }

def task_summary():
    script = ROOT_DIR / 'notebooks/analyze_neck_result.py'
    output = RESULT_DIR / 'summary.html'
    file_dep = [
        RESULT_DIR / 'postprocess/measure_test_reference.ndjson',
        RESULT_DIR / 'postprocess/measure_test_ai.ndjson',
    ]

    # deps of draw phony
    # draw_phony = RESULT_DIR / AI_SUBDIR / 'draw.done'
    # file_dep.append(draw_phony)
    draw_phony = (RESULT_DIR / 'postprocess/reference/draw.done').absolute()
    file_dep.append(draw_phony)

    ai_draw_phony = (RESULT_DIR / 'auto/postprocess/draw.done').absolute()
    file_dep.append(ai_draw_phony)

    # title = '{{"title":"neck summary"}}'
    # c1 = f"jupytext --to ipynb --output - {script} --update-metadata '{title}'"
    c1 = f"jupytext --to ipynb --output - {script}"
    c2 = f'jupyter nbconvert --stdin --to html --execute --no-input --output {output}'

    action = Pipeline().pipe(c1).pipe(c2).runner(text=True)
    return {
        'actions': [action],
        'file_dep': file_dep,
        'targets': [output],
    }


def task_extract_measurements():
    html_move_dir = RESULT_DIR / 'auto/postprocess/draw'
    html_move_dir.mkdir(parents=True, exist_ok=True)

    for fold_dir in FOLD_DIRS:
        measure_ndjson = fold_dir / AI_SUBDIR / 'raw_measurements.ndjson'
        file_dep = fold_dir / AI_SUBDIR / 'deepscol_keypoints.ndjson'

        html_dir = fold_dir / 'result'

        def action():
            all_jsons = []
            for html_file in html_dir.glob('*.html'):
                json_str = extract_measurements_from_html(html_file)
                all_jsons.append((json_str, html_file))
                # shutil.move(str(html_file), html_move_dir / html_file.name)
            with open(measure_ndjson, 'w') as f:
                for json_str, html_file in all_jsons:
                    obj = json.loads(json_str)
                    obj_line = {'content': obj, 'filename': html_file.with_suffix('.json').name}
                    json_str = json.dumps(obj_line, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_str + '\n')

        yield {
            'name': fold_dir.name,
            'actions': [action],
            'file_dep': [file_dep],
            'targets': [measure_ndjson],
        }


def task_apply_model():
    # use fold/0 for external test
    for fold_dir in FOLD_DIRS:
        fold_dir.mkdir(parents=True, exist_ok=True)
        # create path file
        path_file = fold_dir / 'test_paths.txt'

        jq_command = '.content.imagePath'

        ndjson_filename = fold_dir / 'test.ndjson'
        action = Pipeline().pipe(['cat', str(ndjson_filename)]).pipe(['jq', '-r', jq_command])
        action = action.runner(text=True, output_file=path_file)
        yield {
            'name': 'extract-path',
            'actions': [action],
            'file_dep': [ndjson_filename],
            'targets': [path_file],
        }

        # apply model
        outdir = fold_dir / 'result'
        outdir.mkdir(parents=True, exist_ok=True)
        if EXTERNAL_TEST:
            model_dir = os.environ.get('MODEL_DIR')
        else:
            model_dir = fold_dir
        assert model_dir is not None
        model_dir = Path(model_dir)
        logger.debug(f'MODEL_DIR: {model_dir}')
        model_file = model_dir / "model.onnx"
        asm_file = model_dir / TGT_PCA_JSON
        asm_config = SRC_DIR / 'conf/asm_config_neck.toml'
        os.environ['DS__SIZE__IMAGE_RESIZE'] = "800,800"
        os.environ['DS__SESSION__INTRA_THREADS'] = "4"
        # set LD_LIBRARY_PATH to include ONNX Runtime library
        # directory containing `deepscol`
        which_deepscol = os.popen('which deepscol').read().strip()
        deepscol_dir = Path(which_deepscol).parent
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_library_path = f"{deepscol_dir}:{ld_library_path}" if ld_library_path else str(deepscol_dir)
        os.environ['LD_LIBRARY_PATH'] = new_ld_library_path
        # set LD_LIBRARY_PATH for cuda libraries if using cuda
        which_nvcc = os.popen('which nvcc').read().strip()
        if which_nvcc:
            cuda_dir = Path(which_nvcc).parent.parent
            for lib_dir in ['lib64', 'lib']:
                cuda_lib_dir = cuda_dir / lib_dir
                if cuda_lib_dir.exists():
                    os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_dir}:{os.environ['LD_LIBRARY_PATH']}"
                    break
            # onnx runtime provider
            which_deepscol = os.popen('which deepscol').read().strip()
            os.environ['LD_LIBRARY_PATH'] = f"{Path(which_deepscol).parent}:{os.environ['LD_LIBRARY_PATH']}"

        action = [
            'deepscol',
            'batch',
            path_file,
            '--model',
            model_file,
            '--asm',
            f'{asm_file},{asm_config}',
            '--direction',
            'neck-lateral',
            '--labelme',
            '--html',
            '--heatmap',
            '--accelerators',
            'cuda',
            '--inference-workers',
            '4',
            outdir,
        ]
        logger.info(f'Running command: {" ".join([str(a) for a in action])}')

        outfile = fold_dir / AI_SUBDIR / 'deepscol_keypoints.ndjson'
        outfile.parent.mkdir(parents=True, exist_ok=True)
        action2 = (
            Pipeline()
            .pipe(['lmrs', 'ndjson', '--parent', 'remove', str(outdir)])
            .pipe(CMD_ROTATE) # deepscol (unintentionally) sort points by y coordinate. rotate, reorder, and rotate back to original orientation to ensure vertebral corners are sorted/labeled correctly. workaround for deepscol v0.1.2
            .pipe(task_utils.CMD_SORT_VERTEBRAL_CORNERS)
            .pipe(CMD_SORT_LAMINA)
            .pipe(CMD_ROTATE_INV)
            .runner(text=True, output_file=outfile)
        )

        yield {
            'name': fold_dir.name,
            'actions': [action, action2],
            'file_dep': [path_file, model_file, asm_file],
            'targets': [outfile],
        }

def task_normalize_vertebrae():
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / AI_SUBDIR / 'deepscol_keypoints.ndjson'
        midfile = fold_dir / AI_SUBDIR / 'native_deepscol_keypoints.ndjson'
        action1 = create_conv_action(infile, midfile)
        outfile = fold_dir / AI_SUBDIR / 'normalized_deepscol_keypoints.ndjson'
        if NECK_STUDY_POSES.exists():
            action2 = _normalize_vertebral_shape(midfile, NECK_STUDY_POSES, outfile)
        else:
            logger.warning(f"NECK_STUDY_POSES {NECK_STUDY_POSES} does not exist. Skipping shape normalization.")
            action2 = ['cp', midfile, outfile]
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action1, action2],
            'file_dep': [infile],
            'targets': [outfile],
        }


def task_draw_ai():
    html_dir = RESULT_DIR / 'auto/postprocess/draw'
    phony = (html_dir / '../draw.done').absolute()
    phony_action = (make_phony, [phony])

    if NECK_STUDY_POSES.exists():
        infiles = []
        for fold_dir in FOLD_DIRS:
            infile = fold_dir / AI_SUBDIR / 'normalized_deepscol_keypoints.ndjson'
            infiles.append(infile)
        
        config_dir = SRC_DIR / '../configs'
        line_colors = config_dir / 'default_line_colors_label.csv'
        label_colors = config_dir / 'default_label_colors.yaml'
        draw_cmd = [
            'scolrs',
            'svg-ndjson',
            '--html',
            '--label-colors',
            label_colors,
            '--line-colors',
            line_colors,
            '--resize',
            f'{SVG_IMAGE_SIZE}x{SVG_IMAGE_SIZE}',
            "-",
            html_dir,
            'neck',
        ]
        for hide in HIDES:
            draw_cmd += ['--hide', hide]

        cat_cmd = ['cat', *infiles]
        pipeline = Pipeline().pipe(cat_cmd).pipe(draw_cmd).runner()

        yield {
            'name': 'ai-draw',
            'actions': [pipeline, phony_action],
            'file_dep': infiles,
            'targets': [phony],
        }
    else:
        logger.warning(f"NECK_STUDY_POSES {NECK_STUDY_POSES} does not exist. Skipping AI draw task.")
        def action():
            html_move_dir = html_dir
            for fold_dir in FOLD_DIRS:
                for html_file in (fold_dir / 'result').glob('*.html'):
                    shutil.copyfile(str(html_file), html_move_dir / html_file.name)
    
        yield {
            'name': 'ai-draw',
            'actions': [action, phony_action],
            'file_dep': [],
            'targets': [phony],
        }


def task_measure_ai():
    for fold_dir in FOLD_DIRS:
        infile = fold_dir / AI_SUBDIR / 'normalized_deepscol_keypoints.ndjson'
        outfile = fold_dir / AI_SUBDIR / 'measurements.ndjson'
        action = ['scolrs', 'measure', infile, outfile, 'neck']
        yield {
            'name': f'{fold_dir.name}',
            'actions': [action],
            'file_dep': [infile],
            'targets': [outfile],
        }