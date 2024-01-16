import argparse
import json
import sys


parser = argparse.ArgumentParser(description='Convert flattened data to structured ({content, filename}) data')
parser.add_argument('input', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input ndjson filename')
args = parser.parse_args()
for line in args.input:
    data = json.loads(line)
    filename = data.pop('filename')
    structured = dict(content=data, filename=filename)
    print(json.dumps(structured))
