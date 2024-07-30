import json
from pathlib import Path
import functools
import argparse

from detectron2.data import DatasetCatalog, MetadataCatalog


from szkspyne.labelme_utils import load_ndjson

parser = argparse.ArgumentParser(description='Train Detectron2.')
parser.add_argument('input', help='Input directory')
parser.add_argument('output', help='Output directory')
parser.add_argument(
    "--opts",
    help="Modify config options using the command-line 'KEY VALUE' pairs",
    default=[],
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

DATA_DIR = Path(args.input)
OUTPUT_DIR = Path(args.output)

DATASET_PREFIX = 'mydata'

with open(DATA_DIR / 'metadata.json') as f:
    dataset_metadata = json.load(f)
dataset_names = {}
for set_name in ['train', 'val', 'test']:
    ndjson_filename = DATA_DIR / f'{set_name}.ndjson'
    if not ndjson_filename.exists():
        print(f'{ndjson_filename} does not exist. Skipping.')
        continue
    dataset_name = f"{DATASET_PREFIX}_{set_name}"
    dataset_names[set_name] = dataset_name
    DatasetCatalog.register(dataset_name, functools.partial(load_ndjson, ndjson_filename))
    MetadataCatalog.get(dataset_name).set(**dataset_metadata)


from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo

BIG_NUMBER = int(5000)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (dataset_names['train'],)
cfg.DATASETS.TEST = (dataset_names['val'],)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"  # Let training initialize from model zoo
# cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = 'smooth_l1'
cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = 'ciou'
# cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
# cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = BIG_NUMBER
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset_metadata['thing_classes'])
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(dataset_metadata['keypoint_names'])
# cfg.TEST.EVAL_PERIOD = 100
cfg.merge_from_list(args.opts)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cfg.OUTPUT_DIR = str(OUTPUT_DIR)
with open(OUTPUT_DIR / 'cfg.yaml', 'w') as f:
    f.write(cfg.dump())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
