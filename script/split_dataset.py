import sys
import argparse
import json
from pathlib import Path
import random
from logzero import logger


def write_ndjson(data, filename):
    with open(filename, "w") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split dataset.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input ndjson file",
        nargs="+",
    )
    parser.add_argument(
        "--output",
        type=Path,
        nargs='+',
        help="Output ndjson files",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ratio",
        type=float,
        nargs='+',
        help="Split ratio",
    )
    group.add_argument(
        "--ids",
        type=Path,
        nargs='+',
        help="Split based on ids in the input files",
    )
    parser.add_argument(
        "--no-shuffle",
        help="Do not shuffle the data",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=42,
    )
    args = parser.parse_args()

    # read the input ndjson files
    data = []
    for filename in args.input:
        with open(filename) as f:
            ndjson = [json.loads(line) for line in f]
        data.extend(ndjson)
    logger.info(f"Total data: {len(data)}")


    if args.ratio:
        # validate the number of output files and split ratio
        if len(args.output) != len(args.ratio):
            print("Number of output files and split ratio must be equal.")
            return 1

        # normalize the ratio
        total = sum(args.ratio)
        split_ratio = [x / total for x in args.ratio]
        logger.info(f"Split ratio: {split_ratio}")

        if not args.no_shuffle:
            logger.info("Shuffling data")
            random.seed(args.seed)
            random.shuffle(data)

        # split the dataset
        n = len(data)

        split_sizes = [int(n * ratio) for ratio in split_ratio]
        # cumsum
        split_points = [sum(split_sizes[:i]) for i in range(1, len(split_sizes))]
        split_points = [0] + split_points + [n]
        logger.debug(f"Split points: {split_points}")

        total = 0

        for i in range(len(args.output)):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            split_data = data[start_idx:end_idx]
            logger.info(f"Split {i + 1}: {len(split_data)}")
            write_ndjson(split_data, args.output[i])
            total += len(split_data)

        assert total == n

    elif args.ids:
        if len(args.output) != len(args.ids):
            print("Number of output files and ids files must be equal.")
            return 1
        # read the ids from the input files
        total = 0
        for (filename, output_filename) in zip(args.ids, args.output):
            logger.info(f"Reading ids from {filename}")
            with open(filename) as f:
                ids = [line.strip() for line in f]
            ids = set(ids)
            logger.info(f"Total ids: {len(ids)}")
            # filter the data based on the ids
            filtered_data = [obj for obj in data if Path(obj["file_name"]).stem in ids]
            logger.info(f"Filtered data: {len(filtered_data)}")

            # write the filtered data to the output file
            logger.info(f"Writing to {output_filename}")
            write_ndjson(filtered_data, output_filename)
            total += len(filtered_data)
        
        assert total == len(data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
