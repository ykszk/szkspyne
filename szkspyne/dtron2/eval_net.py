#%%
from pathlib import Path
import json

from szkspyne.labelme_utils import load_ndjson

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog

test_dir = Path('../../../result/implant20240703-2')

with open(test_dir / 'metadata.json') as f:
    dataset_metadata = json.load(f)

dataset_name = 'mydata_test'
DatasetCatalog.register(dataset_name, lambda: load_ndjson(test_dir / 'test.ndjson'))
MetadataCatalog.get(dataset_name).set(**dataset_metadata)
for set_name in ['train', 'val', 'test']:
    MetadataCatalog.get(f'mydata_{set_name}').set(**dataset_metadata)
# MetadataCatalog.get(dataset_name).pop('json_file', None)

cfg = get_cfg()
cfg.merge_from_file(str(test_dir / 'cfg.yaml'))
# cfg.MODEL.DEVICE='cpu'
cfg.MODEL.WEIGHTS = str(test_dir / 'model_final.pth')
#%%

# cfg.DATASETS.pop('TRAIN', None)
# cfg.DATASETS.pop('TEST', None)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
# %%
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator(dataset_name, output_dir= test_dir / "test", tasks=['bbox'], allow_cached_coco=False)
val_loader = build_detection_test_loader(cfg, dataset_name)
print(inference_on_dataset(predictor.model, val_loader, evaluator))

# %%
