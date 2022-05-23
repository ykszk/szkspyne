#%%
from pathlib import Path
import json
import numpy as np

LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']

def kps2labelme(kps, image_path, image_hw, label_names):
    json_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_hw[0],
        "imageWidth": image_hw[1]
    }
    shapes = json_data['shapes']
    assert len(label_names) == len(kps)
    for kps, label in zip(kps, label_names):
        for kp in kps:
            shapes.append({
                "label": label,
                "points": [[kp[1], kp[0]]],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            })
    return json_data


if False:
    image_path = Path('tmpx.png').name
    image_shape = [768, 768]
    kps_filename = '../points.json'
    json_filename = '../tmp_output.json'
    with open(kps_filename) as f:
        kps = json.load(f)
    json_data = kps2labelme(image_path, image_shape, kps, LABEL_NAMES)
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
