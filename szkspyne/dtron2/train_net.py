import json
from pathlib import Path
from typing import Union, Dict, List
import functools
import argparse

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

def load_ndjson(filename: Union[str, Path]) -> List[Dict]:
    ndjson: List[Dict] = []
    with open(filename) as f:
        for line in f:
            ndjson.append(json.loads(line))
    return ndjson

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

DATASET_PREFIX = 'scoliosis'
dataset_names = {}
for train_or_val in ['train', 'val']:
    dataset_name = f"{DATASET_PREFIX}_{train_or_val}"
    dataset_names[train_or_val] = dataset_name
    ndjson_filename = DATA_DIR / f'{train_or_val}.ndjson'
    DatasetCatalog.register(dataset_name, functools.partial(load_ndjson, ndjson_filename))
    MetadataCatalog.get(dataset_name).set(thing_classes = ["vertebra"],
                                        keypoint_names=['TL', 'TR', 'BL', 'BR'],
                                        keypoint_flip_map=[('TL','TR'), ('BL','BR')])
    

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
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = BIG_NUMBER    
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vertebrae)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4
# cfg.TEST.EVAL_PERIOD = 100
cfg.merge_from_list(args.opts)

cfg.OUTPUT_DIR = str(OUTPUT_DIR)
with open(OUTPUT_DIR/'cfg.yaml', 'w') as f:
    f.write(cfg.dump())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()