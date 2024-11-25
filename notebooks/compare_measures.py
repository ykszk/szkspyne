import argparse
from pathlib import Path
import sys
from typing import TextIO
from szkspyne.utils.scolrs import MeasureResultLine

def read_ndjson(file_path):
    with open(file_path, 'r') as f:
        ndjson =  [MeasureResultLine.model_validate_json(line) for line in f]
    return {line.filename: line.content for line in ndjson}

def main(ref_file, ai_file, output: TextIO):
    ref_data = read_ndjson(ref_file)
    ai_data = read_ndjson(ai_file)
    assert ref_data.keys() == ai_data.keys()

    for filename in ref_data.keys():
        ref = ref_data[filename]
        ai = ai_data[filename]
        diff = ref.difference(ai)
        line = MeasureResultLine(filename=filename, content=diff)
        output.write(line.model_dump_json() + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare measurement results.')
    parser.add_argument('ref_file', type=Path, help='Path to the reference ndjson file')
    parser.add_argument('ai_file', type=Path, help='Path to the AI ndjson file')
    parser.add_argument('output', type=argparse.FileType('w'), default=sys.stdout, nargs='?', help='Path to the output ndjson file (optional)')

    args = parser.parse_args()
    main(args.ref_file, args.ai_file, args.output)