import sys
import argparse
import json
from pathlib import Path

from logzero import logger
from szkmipy import mhd


def main():
    parser = argparse.ArgumentParser(description='Split mha result from exp into subsets based on flag values.')
    parser.add_argument('mha', help='Input mha filename. (y_pred.mha)', type=Path)
    parser.add_argument('ndjson', help='Input ndjson filename with flags. (test.ndjson)', type=Path)
    parser.add_argument('output', help='Output directory', type=Path)
    parser.add_argument(
        '--flag', help='Flag name to split. (default: %(default)s)', default=['ap_whole', 'lateral_whole'], type=list
    )

    args = parser.parse_args()

    ndjson = [json.loads(line) for line in open(args.ndjson)]
    header = mhd.read_header(args.mha)

    if len(ndjson) != header['DimSize'][-1]:
        logger.error(
            'The number of lines in ndjson and he number of slices does not match. {} != {}'.format(
                len(ndjson), header['DimSize'][-1]
            )
        )
        return 1

    logger.info('Reading mha: %s', args.mha)
    arr, _ = mhd.read(args.mha)

    indexes = {flag: [] for flag in args.flag}
    for i, line in enumerate(ndjson):
        for flag in args.flag:
            if flag in line["content"]["flags"]:
                indexes[flag].append(i)

    # Save mha
    for flag, index_list in indexes.items():
        logger.info('mha %s %s', flag, len(index_list))
        mhd.write(args.output / f'{args.mha.stem}_{flag}{args.mha.suffix}', arr[index_list], header)

    # Save ndjson
    for flag, index_list in indexes.items():
        logger.info('ndjson %s %s', flag, len(index_list))
        with open(args.output / f'{args.ndjson.stem}_{flag}{args.ndjson.suffix}', 'w') as f:
            for i in index_list:
                f.write(json.dumps(ndjson[i]) + '\n')


if __name__ == '__main__':
    sys.exit(main())
