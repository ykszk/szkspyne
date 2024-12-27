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

from szkspyne.labelme_utils import LabelMe
import szkspyne.utils.measure as utils


logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())

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
    lm_json_line: Dict[str, Any],
    json_path: Path,
    resize_param: str,
    svg_size: int,
    label_colors: Dict[str, str],
    line_colors: Dict[str, str],
    image_type: str,
    html_filename_title: tuple[Optional[Path], str],
    hide: Optional[List[str]],
    imagedb: Optional[str],
    extra_point_dict: Dict[str, Dict[str, LabelMe]],
):
    img = None
    if imagedb:
        name = Path(lm_json_line['content']['image_metadata']['path']).stem + '.png'
        img = load_image_from_db(name, imagedb)
        if img:
            resized_shape = pyscol.calc_resize(img.size, resize_param)
            img = img.resize(resized_shape)
            img = np.array(img)
        else:
            logger.warning("Image not found: %s", name)

    draws = []
    draw_param_json = json.dumps(
        dict(radius=POINT_RADIUS, line_width=LINE_WIDTH, font_size='36px', unit_font_size='16px', text_stroke_width=2)
    )
    label_and_point_set = []
    for point_label, data_dict in extra_point_dict.items():
        key = lm_json_line['filename']
        keys = [key, Path(key).name]
        for k in keys:
            if k in data_dict:
                key = k
                break
        else:
            logger.warning("Extra points not found for %s", key)
            continue

        if key in data_dict:
            extra_data = data_dict[key]
            extra_points = []
            for point in extra_data.shapes:
                if point.shape_type == 'point':
                    extra_points.append(point.points[0])
            arr = np.stack(extra_points)
            assert arr.ndim == 2, f"Extra points should be 2D array, but got {arr.ndim}D"
            label_and_point_set.append((point_label, arr))
        else:
            logger.warning("Extra points not found for %s", key)

    lm_json = json.dumps(lm_json_line['content'])

    image_aspect_ratio = (
        lm_json_line['content']['image_metadata']['width'] / lm_json_line['content']['image_metadata']['height']
    )
    if image_aspect_ratio > 1:
        svg_wh = (svg_size, int(svg_size / image_aspect_ratio))
    else:
        svg_wh = (int(svg_size * image_aspect_ratio), svg_size)

    if image_type == 'ap':
        svg = pyscol.draw_coronal(
            lm_json,
            json_path,
            draws,
            hide,
            draw_param_json,
            label_colors,
            line_colors,
            resize_param,
            svg_wh,
            img,
            label_and_point_set,
        )
    elif image_type == 'lateral':
        svg = pyscol.draw_sagittal(
            lm_json,
            json_path,
            draws,
            hide,
            draw_param_json,
            label_colors,
            line_colors,
            resize_param,
            svg_wh,
            img,
            label_and_point_set,
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
    parser.add_argument('--hide', help='Hide specified drawings', nargs='*', default=None)
    parser.add_argument('--resize', help='Resize parameter. default: %(default)s', type=str, default="1024x1024")
    parser.add_argument('--svg_size', help='SVG size', type=int, default=1024)
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
        lm_json_line = {'content': lm_json, "filename": args.input.name}
        measure_json(
            lm_json_line,
            args.input,
            args.resize,
            args.svg_size,
            label_colors,
            line_colors,
            args.image_type,
            (args.html, args.title),
            args.hide,
            args.imagedb,
            {},
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
    sub_batch.add_argument(
        '--extra_points',
        help='Extra points (in labelme format) ndjson filename',
        metavar='<extra_points>',
        nargs='*',
    )
    sub_batch.add_argument(
        '--extra_labels',
        help='Labels for extra points',
        metavar='<extra_labels>',
        nargs='*',
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

        if len(args.extra_points) != len(args.extra_labels):
            print('Extra points and labels must have the same length')
            sys.exit(1)

        extra_point_dict = {}
        for el, ep in zip(args.extra_labels, args.extra_points):
            with open(ep) as f:
                ndjson = [json.loads(line) for line in f]
            extra_dict = {obj['filename']: LabelMe.model_validate(obj['content']) for obj in ndjson}
            extra_point_dict[el] = extra_dict
            # add colors
            if 'reference' in el:
                label_colors[el] = 'cyan'
                line_colors[el] = 'black'
            else:
                label_colors[el] = 'white'
                line_colors[el] = 'black'

        def do_ndjson(lm_data_line: Any):
            stem = Path(lm_data_line['content']['image_metadata']['path']).stem
            html = outdir / (stem + '.html') if args.html else None
            measure_json(
                lm_data_line,
                args.input,
                args.resize,
                args.svg_size,
                label_colors,
                line_colors,
                args.image_type,
                (html, '{} - {}'.format(stem, args.title)),
                args.hide,
                args.imagedb,
                extra_point_dict,
            )

        if args.input == '-':
            ndjson = [json.loads(line) for line in sys.stdin]
        else:
            with open(args.input) as f:
                ndjson = [json.loads(line) for line in f]
        Parallel(n_jobs=args.workers)(delayed(do_ndjson)(data) for data in tqdm.tqdm(ndjson))

    sub_batch.set_defaults(handler=command_batch)

    args = parser.parse_args()

    if not hasattr(args, 'handler'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
