#%%
from logzero import logger
import yaml
with open('../../tools/bin/config.yaml') as f:
    color_data = yaml.safe_load(f)
label_colors = color_data['label_colors']

from pathlib import Path
import json

anno_dir = Path('../../data/annotation/')
fns = sorted(anno_dir.glob('**/valid*.ndjson'))
d_ndj = {}
for fn in fns:
    logger.info(fn)
    with open(fn, encoding='utf8') as f:
        ndj = [json.loads(line) for line in f]
    direction = '_'.join(fn.stem.split('_')[-2:])
    d_ndj[direction] = d_ndj.get(direction, []) + ndj

import sys

sys.path.append('../spinex')
import labelme_utils
import io
from base64 import b64encode
import svgwrite
from PIL import Image
import numpy as np


def img2b64(img):
    with io.BytesIO() as bio:
        img.save(bio, format='jpeg')
        binary = bio.getvalue()
    b64 = b64encode(binary)
    str_b64 = b64.decode('ascii')
    return str_b64


from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('template', encoding='utf8'))
t_catalog = env.get_template('catalog.html')
t_img = env.get_template('img.html')
t_legend = env.get_template('legend.html')

img_dir = Path('../../data/jpg_half')


def json2imgpath(filename):
    image_filename = (img_dir /
                      '/'.join(filename.split('_'))).with_suffix('.jpg')
    return image_filename


for direction, ndj in d_ndj.items():
    logger.info(direction)
    body = []
    for data in ndj:
        flags = ', '.join([f for f, b in data['flags'].items() if b])
        img_filename = json2imgpath(data['filepath'].split('/')[-1])

        shapes = labelme_utils.conv2shape_dict(data)
        points = shapes['point']
        img = Image.open(img_filename)
        original_size = img.size
        img.thumbnail((400, 600))
        SCALE = img.size[0] / original_size[0]
        w, h = img.size
        pixel_size = 1
        POINT_RADIUS = 2
        dwg = svgwrite.Drawing('output.svg',
                               size=(w * pixel_size, h * pixel_size),
                               viewBox=('0 0 {} {}'.format(w, h)))
        dwg.add(
            dwg.image('data:image/jpg;base64,' + img2b64(img), size=img.size))
        anno_grp = dwg.add(dwg.g(class_='annotation'))
        for label, pts in points.items():
            grp = anno_grp.add(
                dwg.g(id=label, class_='point', stroke_width=2.0, fill='none'))
            if label not in label_colors:
                continue
            color = '#' + ''.join(
                ['{:02x}'.format(c) for c in label_colors[label]])
            pts = SCALE * np.array(pts).squeeze()
            for pt in pts:
                grp.add(
                    dwg.circle(center=pt.tolist(),
                               r=POINT_RADIUS,
                               stroke=color,
                               fill=color))
        for label, circles in shapes.get('circle', {}).items():
            circles = SCALE * np.array(circles)
            color = '#' + ''.join(
                ['{:02x}'.format(c) for c in label_colors[label]])
            grp = anno_grp.add(
                dwg.g(id=label, class_='circle', stroke_width=2.0,
                      fill='none'))
            for circle in circles:
                grp.add(
                    dwg.circle(center=circle[0].tolist(),
                               r=POINT_RADIUS,
                               stroke=color,
                               fill=color))
                if circle.shape[0] > 1:
                    radius = np.linalg.norm(circle[1] - circle[0])
                    grp.add(
                        dwg.circle(center=circle[0].tolist(),
                                   r=radius,
                                   stroke=color,
                                   fill='none'))
        for label, rectangles in shapes.get('rectangle', {}).items():
            rectangles = SCALE * np.array(rectangles)
            color = '#' + ''.join(
                ['{:02x}'.format(c) for c in label_colors[label]])
            grp = anno_grp.add(
                dwg.g(id=label,
                      class_='rectangle',
                      stroke_width=2.0,
                      fill='none'))
            for rectangle in rectangles:
                grp.add(
                    dwg.rect(rectangle[0],
                             rectangle[1] - rectangle[0],
                             stroke=color,
                             fill='none'))

        svg_str = dwg.tostring()
        body.append(
            t_img.render(svg=svg_str,
                         flags=flags,
                         name=Path(data['filepath']).stem))

    body = '\n'.join(body)
    legend = []
    for label, color in label_colors.items():
        color = '#' + ''.join(
            ['{:02x}'.format(c) for c in label_colors[label]])
        legend.append(t_legend.render(label=label, color=color))
    legend = '\n'.join(legend)
    with open(f'output/catalog_{direction}.html', 'w', encoding='utf8') as f:
        f.write(t_catalog.render(title=direction, legend=legend, main=body))
