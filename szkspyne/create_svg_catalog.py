from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser(
    description='Create catalog from measure_and_draw output.')
parser.add_argument('input', help='Input directory', metavar='<input>')
parser.add_argument('output', help='Output html filename', metavar='<output>')
parser.add_argument('--title', help='html title')
parser.add_argument('-s',
                    '--switch',
                    help='Switch argument',
                    action='store_true')

args = parser.parse_args()

indir = Path(args.input)
output = Path(args.output)
title = output.stem
if args.title:
    title = args.title
outdir = output.parent
count = 0
html_lines = []
for svg_filename in sorted(indir.glob('*.svg')):
    rel_fn = str(svg_filename.relative_to(outdir))
    rel_html = str(svg_filename.relative_to(outdir).with_suffix('.html'))
    caption = svg_filename.stem
    html_lines.append(
        f'<div class="img"><figure><a href="{rel_html}"><figcaption>{caption}</figcaption></a><img src="{rel_fn}"></figure></div>'
    )

style = '''
<style>
img {
    max-height: 512px;
}
body {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
}
</style>
'''
with open(output, 'w') as f:
    f.write(style)
    f.write('<body>')
    f.write('\n'.join(html_lines))
    f.write('</body>')