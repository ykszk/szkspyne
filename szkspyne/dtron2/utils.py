import json

import detectron2
import detectron2.structures
import torch


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        else:
            return super().default(obj)

class BoxesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, detectron2.structures.Boxes):
            return obj.tensor
        else:
            return super().default(obj)

class InstancesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, detectron2.structures.instances.Instances):
            fields = obj.get_fields()
            fields.pop('pred_keypoint_heatmaps')
            fields['image_size'] = obj.image_size
            return fields
        else:
            return super().default(obj)

class PredictionEncoder(TensorEncoder, InstancesEncoder, BoxesEncoder):
    pass