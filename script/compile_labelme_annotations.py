# %%
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from logzero import logger

from szkspyne import labelme_utils


parser = argparse.ArgumentParser(description='Convert labelme jsonl to exp format. Input is read from stdin')
parser.add_argument(
    '--noconv', help='No dataformat conversion. Just concat input and remove duplicates', action='store_true'
)
parser.add_argument('--check_image', help='Check image file existence.', action='store_true')

args = parser.parse_args()


dataset = {}
for line in sys.stdin:
    data = json.loads(line)
    if args.noconv:
        dataset[data['filepath']] = data
    else:
        shape_data = labelme_utils.conv2shape_dict(data)
        shape_data['filepath'] = data['filepath']
        if 'image_type' in data:
            shape_data['image_type'] = data['image_type']
        shape_data['flags'] = [k for k, v in data['flags'].items() if v]
        if args.check_image and not os.path.exists(data['filepath']):
            logger.warning('Skip entry with no image: %s', data['filepath'])
            continue
        for t_copy in ['imageHeight', 'imageWidth']:
            shape_data[t_copy] = data[t_copy]
        dataset[data['filepath']] = shape_data

for data in dataset.values():
    print(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
