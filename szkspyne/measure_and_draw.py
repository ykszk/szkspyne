import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import logzero
import numpy as np
import yaml
from logzero import logger
from PIL import Image
import pyscol

import szkspyne.utils.measure as utils


logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())

LINE_WIDTH = 2.0
POINT_RADIUS = 2.0
PERP_BOX_SIZE = (20, 20)  # perpendicular sign


def load_line_colors():
    return utils.read_default_line_colors()


def color_to_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def load_label_colors(filename, to_hex=True) -> Dict[str, str]:
    with open(filename) as f:
        color_data = yaml.safe_load(f)
    label_colors = color_data['label_colors']
    if to_hex:
        label_colors = {k: color_to_hex(v) for k, v in label_colors.items()}
    return label_colors


def load_image_from_db(filename, imagedb) -> Optional[Image.Image]:
    with sqlite3.connect(imagedb) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT binary FROM image WHERE name = ?", (filename,))
        binary_data = cursor.fetchone()
    if binary_data:
        binary_data = binary_data[0]
        file_jpgdata = BytesIO(binary_data)
        img = Image.open(file_jpgdata)
        return img
    return None


def measure_json(
    lm_json: str,
    json_path: Path,
    resize_param: str,
    label_colors: Dict[str, str],
    line_colors: Dict[str, str],
    image_type: str,
    html_filename_title: tuple[Optional[Path], str],
    hide: List[str],
    imagedb: Optional[str],
):

    d_visibility = defaultdict(lambda: 'visible')
    d_visibility.update({gid: 'hidden' for gid in hide})

    img = None
    if imagedb:
        lm_data = json.loads(lm_json)
        name = Path(lm_data['image_metadata']['path']).stem + '.png'
        img = load_image_from_db(name, imagedb)
        if img:
            resized_shape = pyscol.calc_resize(img.size, resize_param)
            img = img.resize(resized_shape)
            img = np.array(img)
        else:
            logger.warning("Image not found: %s", name)

    draws = []
    draw_param_json = "{}"

    if image_type == 'ap':
        svg = pyscol.draw_coronal(
            lm_json, json_path, draws, hide, draw_param_json, label_colors, line_colors, resize_param, img
        )
    elif image_type == 'lateral':
        svg = pyscol.draw_sagittal(
            lm_json, json_path, draws, hide, draw_param_json, label_colors, line_colors, resize_param, img
        )
    else:
        raise ValueError(f"Unknown image type: {image_type}")

    html_filename, html_title = html_filename_title
    if html_filename:
        html = pyscol.wrap_in_html(svg, html_title)
        with open(html_filename, 'w') as f:
            f.write(html)


def main():
    parser = argparse.ArgumentParser(description='Measure and draw.')
    parser.add_argument('-c', '--config', help='Config yaml filename', required=True)
    parser.add_argument('--title', help='html title', default='Scoliosis Measuaraments')
    parser.add_argument('--image_type', help='Image type', choices=['ap', 'lateral'], default='ap')
    parser.add_argument('--imagedb', help='image database filename for heatmap overlay')
    parser.add_argument('--hide', help='Hide specified drawings', nargs='*', default=[])

    parser.add_argument('--resize', help='Resize parameter. default: %(default)s', type=str, default="1024x1024")
    # parser.add_argument('--slopes', help='Add slopes to measuremens json', action='store_true')

    subparsers = parser.add_subparsers()
    sub_single = subparsers.add_parser('single', help="measure and draw from given file")
    sub_single.add_argument('input', help='Input json filename', metavar='<input>', type=Path)
    sub_single.add_argument('--html', help='Save html')

    def command_single(args):
        label_colors = load_label_colors(args.config)
        line_colors = load_line_colors()
        with open(args.input) as f:
            lm_json = f.read()
        measure_json(
            lm_json,
            args.input,
            args.resize,
            label_colors,
            line_colors,
            args.image_type,
            (args.html, args.title),
            args.hide,
            args.imagedb,
        )

    sub_single.set_defaults(handler=command_single)

    sub_batch = subparsers.add_parser('batch', help="measure and draw from given directory")
    sub_batch.add_argument('input', help='Input ndjson file "-" for stdin in ndjson format')
    sub_batch.add_argument('output', help='Output directory')
    sub_batch.add_argument('--html', help='Save html', action='store_true')
    sub_batch.add_argument('--image', help='Save image', metavar='ext', choices=['.svg', '.png'])
    sub_batch.add_argument(
        '--workers', help='The number of concurrent workers. Use all available cores by default', type=int, default=0
    )

    def command_batch(args):
        if not args.html and not args.image:
            print('At least one output (html, image, or measure) argument is required.')
            sys.exit(1)
        outdir = Path(args.output)
        label_colors = load_label_colors(args.config)
        line_colors = load_line_colors()
        if args.workers <= 0:
            import os

            args.workers = os.cpu_count()
            logger.info("Use %d cores", args.workers)
        import tqdm
        from joblib import Parallel, delayed

        def do_ndjson(lm_data: Any):
            lm_json_str = json.dumps(lm_data)
            stem = Path(lm_data['image_metadata']['path']).stem
            html = outdir / (stem + '.html') if args.html else None
            measure_json(
                lm_json_str,
                args.input,
                args.resize,
                label_colors,
                line_colors,
                args.image_type,
                (html, '{} - {}'.format(stem, args.title)),
                args.hide,
                args.imagedb,
            )

        if args.input == '-':
            ndjson = [json.loads(line) for line in sys.stdin]
        else:
            with open(args.input) as f:
                ndjson = [json.loads(line) for line in f]
        Parallel(n_jobs=args.workers)(delayed(do_ndjson)(data['content']) for data in tqdm.tqdm(ndjson))

    sub_batch.set_defaults(handler=command_batch)

    args = parser.parse_args()

    if not hasattr(args, 'handler'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
