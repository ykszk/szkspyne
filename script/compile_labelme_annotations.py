import argparse
import json
import os
import sys
from typing import Dict

from logzero import logger

from szkspyne import labelme_utils


parser = argparse.ArgumentParser(description='Convert labelme jsonl to exp format. Input is read from stdin')
parser.add_argument('input', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input ndjson filename')
parser.add_argument('--check_image', help='Check image file existence.', action='store_true')

args = parser.parse_args()


dataset: Dict[str, labelme_utils.ShapeDictLine] = {}
for line in args.input:
    data = labelme_utils.LabelMeLine.model_validate_json(line)
    shape_data = data.into_shape_dict_line()
    if args.check_image and not os.path.exists(data.content.imagePath):
        logger.warning('Skip entry with no image: %s', data.content.imagePath)
        continue
    dataset[data.filename] = shape_data

for data in dataset.values():
    print(data.model_dump_json())
