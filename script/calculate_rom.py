import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from szkspyne.type.neck_study import SortedFiles, Studies, UnsortedFiles
from szkspyne.utils import scolrs

parser = argparse.ArgumentParser(description='Calculate ROM from reference and AI measurements')
parser.add_argument('study_pose_file', type=Path, help='Path to the study pose file')
parser.add_argument('measures', type=Path, help='Path to the measurement ndjson file')
parser.add_argument('output', type=Path, help='Path to the output CSV file')
args = parser.parse_args()

study_poses = Studies.model_validate_json(args.study_pose_file.read_text())
print(f'Loaded {len(study_poses.root)} study poses')
rom_index = ['C2/3', 'C3/4', 'C4/5', 'C5/6', 'C6/7', 'C7/T1']

def load_as_measures(filename: str, exclude = None) -> List[scolrs.MeasureResultLine]:
    lines = scolrs.MeasureResultLine.from_ndjson(filename)
    new_lines = []
    # strip parent directory form .filename
    for line in lines:
        if exclude and Path(line.filename).stem in exclude:
            continue
        line.filename = Path(line.filename).name
        new_lines.append(line)
    return new_lines


measures = load_as_measures(args.measures, [])

measure_dict = {m.filename: m for m in measures}
print(f'Loaded {len(measures)} measures')

roms = {}

for study_id, poses in study_poses.root.items():
    if isinstance(poses, UnsortedFiles):
        continue
    elif isinstance(poses, SortedFiles):
        pass
    else:
        continue
    if not (poses.flexion and poses.extension): # either flexion or extension is missing
        continue
    def extract_rom_measure(measure_dict, poses):
        flexion_measure = measure_dict.get(poses.flexion, None)
        extension_measure = measure_dict.get(poses.extension, None)
        if flexion_measure is None:
            print(f'Could not find flexion measure for study {study_id} with filename {poses.flexion}')
            return None
        if extension_measure is None:
            print(f'Could not find extension measure for study {study_id} with filename {poses.extension}')
            return None
        flex_measure = flexion_measure.content.measurements
        ext_measure = extension_measure.content.measurements
        # get `InterVertebralAngle_?`s
        keys = [f'InterVertebralAngle_{i}' for i in range(1, 7)]
        flex_angles = np.array([flex_measure[k].Ok for k in keys])
        ext_angles = np.array([ext_measure[k].Ok for k in keys])

        rom = ext_angles - flex_angles
        return rom
    rom = extract_rom_measure(measure_dict, poses)
    if rom is not None:
        roms[study_id] = rom

rom_df = pd.DataFrame.from_dict(roms, orient='index', columns=rom_index)
rom_df.to_csv(args.output)
