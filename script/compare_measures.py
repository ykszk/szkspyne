import argparse
from pathlib import Path
import sys
from typing import TextIO
from szkspyne.utils.scolrs import MeasureResultLine


def read_ndjson(file_path, basename):
    with open(file_path, 'r') as f:
        ndjson = [MeasureResultLine.model_validate_json(line) for line in f]
    if basename:
        return {Path(line.filename).name: line.content for line in ndjson}
    return {line.filename: line.content for line in ndjson}


def main():
    parser = argparse.ArgumentParser(description='Compare measurement results.')
    parser.add_argument('ref_file', type=Path, help='Path to the reference ndjson file')
    parser.add_argument('ai_file', type=Path, help='Path to the AI ndjson file')
    parser.add_argument(
        'output',
        type=argparse.FileType('w'),
        default=sys.stdout,
        nargs='?',
        help='Path to the output ndjson file (optional)',
    )
    parser.add_argument('--dataframe', help='Output as a pandas DataFrame')
    parser.add_argument('--basename', help='Use basename for filename', action='store_true')

    args = parser.parse_args()

    ref_file = args.ref_file
    ai_file = args.ai_file
    output: TextIO = args.output

    ref_data = read_ndjson(ref_file, args.basename)
    ai_data = read_ndjson(ai_file, args.basename)
    assert ref_data.keys() == ai_data.keys(), f'Keys are different: {len(ref_data.keys())} != {len(ai_data.keys())}'

    lines = []
    for filename in ref_data.keys():
        ref = ref_data[filename]
        ai = ai_data[filename]
        diff = ref.difference(ai)
        line = MeasureResultLine(filename=filename, content=diff)
        output.write(line.model_dump_json() + '\n')
        if args.dataframe:
            lines.append(line)

    if args.dataframe:
        df = MeasureResultLine.to_dataframe(lines)
        df.to_csv(args.dataframe, index=True)


if __name__ == "__main__":
    main()
