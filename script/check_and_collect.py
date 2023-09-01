# %%
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import lmrspy
import numpy as np
import tqdm


# https://github.com/ykszk/labelme-rs

parser = argparse.ArgumentParser(
    description='Create ndjson from a labelme directory.')
parser.add_argument('input', help='Input data directory', metavar='<input>')
parser.add_argument('output', help='Output directory', metavar='<output>')
parser.add_argument('--annotators',
                    help='List of annotators. default: %(default)s',
                    default=['藤森', '中嶋', '蟹江'])
parser.add_argument(
    '--ruledir',
    help=
    'Directory containing rules files such as "rule-spine.txt". default: %(default)s',
    default='rules')
args = parser.parse_args()

data_dir = Path(args.input)
annotators = set(args.annotators)
outdir = Path(args.output)
ruledir = Path(args.ruledir)


def save_ndjson(filename: Union[Path, str], json_list: List[dict]):
    with open(filename, 'w') as f:
        for j in json_list:
            json.dump(j, f, ensure_ascii=False)
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
for fn in ['L4','L5','L6','congenital','spine', 'lateral-spine']:
    rule_dict[fn] = load_rule_file(ruledir / f'rule-{fn}.txt')

rule_sets = {}
for rule in ['L4', 'L5', 'L6', 'congenital']:
    rule_set = rule_dict['spine'] + rule_dict[rule]
    rule_sets[rule] = rule_set

for kind in ['ap_whole', 'lateral_whole']:
    print(kind)
    fns = sorted(data_dir.glob(kind + '/*.json'))
    validators: List[lmrspy.Validator] = []
    for ignore in ['L5', 'L4', 'L6', '先天性側彎']:
        ignores = IGNORES.copy()
        ignores.remove(ignore)
        key=ignore
        if key=='先天性側彎':
            key='congenital'
        rule_set = rule_sets[key]
        if kind=='lateral_whole':
            rule_set = rule_set + rule_dict['lateral-spine']
        validators.append(lmrspy.Validator(rule_set, args.annotators, ignores))

    valid_json = []
    invalid_json = []
    for fn in tqdm.tqdm(fns):
        with open(fn) as f:
            json_str = f.read()
        json_data = json.loads(json_str)
        json_data['filepath'] = str(fn)
        for validator in validators:
            try:
                result = validator.validate_jsons(json_str)
                if result: # passed
                    valid_json.append(json_data)
                    break
                else: # skipped
                    # try next validator
                    continue
            except Exception as e:
                print(fn, e)
                invalid_json.append(json_data)
                continue
    print('Total file count:', len(fns))
    print('Valid:', len(valid_json))
    print('Invalid:', len(invalid_json))
    outname = outdir / ('valid_' + kind + '.ndjson')
    outname.parent.mkdir(parents=True, exist_ok=True)
    save_ndjson(outname, valid_json)
    if len(invalid_json) > 0:
        outname = outdir / ('invalid_' + kind + '.ndjson')
        save_ndjson(outname, invalid_json)
