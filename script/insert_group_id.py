import argparse
import csv
import json
import re
import sys


def main():
    parser = argparse.ArgumentParser(description="Read ndjson from stdin and insert group id.")
    parser.add_argument("--src_key", help="Extract group id from this key. default: %(default)s", default="filepath")
    parser.add_argument("--id_key", help="Insert group id into this key. default: %(default)s", default="group_id")
    parser.add_argument("--map", help="Save resulting key to gorup_id map as a csv.")
    args = parser.parse_args()

    path2id = []
    for line in sys.stdin:
        data = json.loads(line)
        filepath: str = data[args.src_key]
        if filepath.startswith("OSC"):
            group_id = filepath.split("_")[0]
        elif filepath.startswith("CMCAISY"):
            m = re.search(r"\d+", filepath)
            if m is None:
                raise RuntimeError("No number found in: " + filepath)
            group_id = "CMCAISY" + m.group()
        else:
            group_id = re.split(r"-|=|\.|,| |_|yr", filepath)[0]
        path2id.append((filepath, group_id))
        data[args.id_key] = group_id
        print(json.dumps(data, ensure_ascii=False))

    if args.map:
        with open(args.map, mode="w", newline="") as file:
            writer = csv.writer(file)

            for row in path2id:
                writer.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
