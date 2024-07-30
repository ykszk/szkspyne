import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Plot metrics.json of detectron2.')
    parser.add_argument('input', help='metrics.json file.')
    parser.add_argument('output', help='Output directory', nargs='?')

    args = parser.parse_args()

    # lazy load modules
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns

    with open(args.input) as f:
        metrics = [json.loads(line) for line in f]

    df = pd.DataFrame(metrics)
    df.head()

    outdir = Path(args.output) if args.output else Path(args.input).parent

    sns.set_theme(context='notebook', style='whitegrid', font='serif')

    for name, metric_set in [('loss', ['total_loss', 'loss_cls', 'loss_box_reg']), ('accuracy', ['fast_rcnn/cls_accuracy', 'fast_rcnn/fg_cls_accuracy'])]:
        df_melted = df.melt(id_vars='iteration', value_vars = metric_set, var_name='metric', value_name='value')
        sns.lineplot(data=df_melted, x='iteration', y='value', hue='metric')
        plt.gcf().savefig(outdir / f'{name}.png',
                    #   transparent=True,
                    bbox_inches='tight', dpi=200)
        plt.clf()

if __name__ == '__main__':
    sys.exit(main())

