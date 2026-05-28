# %%
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import lmrspy
import numpy as np
import tqdm
from logzero import logger

from szkspyne import labelme_utils

# https://github.com/ykszk/labelme-rs

parser = argparse.ArgumentParser(description='Create ndjson from a labelme directory.')
parser.add_argument('input', help='Input data directory', metavar='<input>')
parser.add_argument('output', help='Output directory', metavar='<output>')
parser.add_argument('--annotators', help='List of annotators. default: %(default)s', default=['藤森', '中嶋', '蟹江'])
parser.add_argument(
    '--ruledir',
    help='Directory containing rules files such as "rule-spine.txt". default: %(default)s',
    default='rules',
)
parser.add_argument('--add_flag', help='Add new flag to the output', type=str, required=False, nargs="*", default=[])
parser.add_argument('--image_parent', help='Parent directory of images', type=Path)
parser.add_argument('--ap', help='Subdirectory for ap_whole images', nargs='*',type=str, default=['ap_whole'])
parser.add_argument('--lateral', help='Subdirectory for lateral_whole images', nargs='*', type=str, default=['lateral_whole'])
args = parser.parse_args()

data_dir = Path(args.input)
annotators = set(args.annotators)
outdir = Path(args.output)
ruledir = Path(args.ruledir)


def save_ndjson(filename: Union[Path, str], json_list: List[Path], flags: List[str], image_parent: Optional[Path]):
    with open(filename, 'w') as f:
        for j in json_list:
            lm_line = labelme_utils.LabelMe.from_file(j).into_labelme_line(str(j))
            for flag in flags:
                lm_line.content.flags[flag] = True
            if image_parent:
                original_path = lm_line.content.imagePath.replace('\\', '/')
                lm_line.content.imagePath = str(image_parent / Path(original_path).name)
            f.write(lm_line.model_dump_json())
            f.write('\n')


# IGNORES = ['除外', 'L4', 'L5', 'L6', '先天性側彎']
IGNORES = ['L4', 'L5', 'L6', '先天性側彎']


def load_rule_file(filename):
    with open(filename) as f:
        rules = f.readlines()
    rules = [r.rstrip() for r in rules]
    rules = [r for r in rules if r]
    return rules


rule_dict = {}
for fn in ['L4', 'L5', 'L6', 'congenital', 'spine', 'lateral-spine']:
    rule_dict[fn] = load_rule_file(ruledir / f'rule-{fn}.txt')

rule_sets = {}
for rule in ['L4', 'L5', 'L6', 'congenital']:
    rule_set = rule_dict['spine'] + rule_dict[rule]
    rule_sets[rule] = rule_set

subdirs = [('ap_whole', subdir) for subdir in args.ap] + [('lateral_whole', subdir) for subdir in args.lateral]
# for kind in ['ap_whole', 'lateral_whole']:
for kind, subdir in subdirs:
    logger.info(f'Processing {subdir} as {kind}')
    fns = sorted(data_dir.glob(subdir + '/*.json'))
    validators: List[lmrspy.Validator] = []
    for ignore in ['L5', 'L4', 'L6', '先天性側彎']:
        ignores = IGNORES.copy()
        ignores.remove(ignore)
        key = ignore
        if key == '先天性側彎':
            key = 'congenital'
        rule_set = rule_sets[key]
        if kind == 'lateral_whole':
            rule_set = rule_set + rule_dict['lateral-spine']
        validators.append(lmrspy.Validator(rule_set, args.annotators, ignores))
    logger.info(f'Loaded {len(validators)} validators for {kind}')
    valid_json: List[Path] = []
    invalid_json: List[Path] = []
    skipped_json: List[Path] = []
    for fn in tqdm.tqdm(fns):
        with open(fn) as f:
            json_str = f.read()
        json_data = json.loads(json_str)
        json_data['filepath'] = str(fn)
        for validator in validators:
            try:
                result = validator.validate_jsons(json_str)
                if result:  # passed
                    valid_json.append(fn)
                    break
                else:  # skipped
                    # try next validator
                    continue
            except Exception as e:
                print(fn, e)
                invalid_json.append(fn)
                break
        else:
            skipped_json.append(fn)

    print('Total file count:', len(fns))
    print('Valid:', len(valid_json))
    print('Invalid:', len(invalid_json))
    print('Skipped:', len(skipped_json))
    if kind==subdir:
        outname = outdir / ('valid_' + kind + '.ndjson')
    else:
        outname = outdir / (f'valid_{kind}_{subdir}.ndjson')
    outname.parent.mkdir(parents=True, exist_ok=True)
    save_ndjson(outname, valid_json, args.add_flag, args.image_parent)
    if len(invalid_json) > 0:
        outname = outdir / ('invalid_' + kind + '.ndjson')
        save_ndjson(outname, invalid_json, [], args.image_parent)
    if len(skipped_json) > 0:
        outname = outdir / ('skipped_' + kind + '.ndjson')
        save_ndjson(outname, skipped_json, [], args.image_parent)
