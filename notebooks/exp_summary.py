# %%
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import japanize_matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from IPython.display import HTML, display
from jinja2 import Environment, FileSystemLoader
from scipy import stats

import szkspyne
from szkspyne.utils.measure import pearsonr_ci


mpl.rcParams["figure.dpi"] = 100

sns.set(context="notebook", style="whitegrid", font="IPAexGothic")

template_directory = Path(szkspyne.__file__).parent / 'template'
env = Environment(loader=FileSystemLoader(template_directory, encoding='utf8'))
t_cb = env.get_template('t_checkbox.html')

result_dir = Path(os.environ["INPUT_DIR"])
include_visualization = not ('NO_VIS' in os.environ)

# %%
from collections import namedtuple


STORED_CURVE = 'STORED_CURVE' in os.environ
ap_filename = 'ap_whole_w_curve_indices.csv' if STORED_CURVE else 'ap_whole.csv'
dir_suffix = os.environ.get("DIR_SUFFIX", '_whole')
# df_birthday = pd.read_csv('../../data/anonymized_birthday.csv')
# dict_sex_bday = {row.PatientID:(row.PatientSex, datetime.strptime(str(row.Birthday), '%Y%m%d')) for _, row in df_birthday.iterrows()}
ImageType = namedtuple("ImageType",
                       ["name", "filename", "angle_cols", "key_cols"])
d_ap = ImageType(
    name="ap",
    filename=ap_filename,
    angle_cols=[
        "PT", "MT", "LT", "T1Tilt", "ClavicleAngle", "PelvicObliquity",
        "SacralObliquity"
    ],
    key_cols=["MT", "PelvicObliquity"],
)

d_lateral = ImageType(
    name="lateral",
    filename="lateral_whole.csv",
    angle_cols=[
        "ThoracicKyphosis",
        "Mid/LowerThoracicKyphosis",
        "ProximalThoracicKyphosis",
        "LumbarLordosis",
        "SacralSlope",
        "PelvicIncidence",
        "PelvicRadiusAngle",
        "L5IncidenceAngle",
        "LumbosacralAngle",
    ],
    key_cols=["ThoracicKyphosis", "LumbarLordosis"],
)

STYLE = '''
<style>
div.sbs {
    display: flex;
    flex-direction: row;
}

div.img {
    margin: 1em;
}
body {
display:flex;
flex-directoin: row;
flex-wrap: wrap;
}
#control {
position: fixed;
right:0;
top:0;
}
</style>
'''
JAVASCRIPT = '''
    <script>
        function change_visibility(cb) {
            const svg_id = cb.id.substring(3);
            const visibility = cb.checked ? "visible" : "hidden";
            const elements = document.getElementsByClassName(svg_id)
			for (const element of elements) {
				element.style.visibility = visibility;
			}
        }

        function change_all(visibility) {
            const elms = document.getElementsByTagName('input');
            console.log(visibility);
            for (elm of elms) {
                elm.checked = visibility == "visible";
                change_visibility(elm);
            }
        }
    </script>
'''

FLAG_COLS = ['術後', 'EOS', 'L4', 'L5']

for d in [d_ap, d_lateral]:
    name, filename, angle_cols, key_cols = d
    filename = str(result_dir / filename)
    display(HTML(f"<h1>{name}</h1>"))
    df = pd.read_csv(filename, encoding="cp932")
    for c in ["除外", "左屈", "右屈", "L6", "先天性側彎"]:
        df = df[np.logical_not(df[c])]
        del df[c]
    display(df.head())

    df_dropped = df.dropna()

    pids = df["filename"].map(lambda e: e.split("_")[0])
    print("# of images", len(df))
    print("# of unique patients", len(set(pids)))

    # sexes, ages = [], []
    for fn in df["filename"]:
        pid = fn.split("_")[0].split("-")[0]
        # pid, study_date = fn.split('_')[:2]
        # study_date = datetime.strptime(study_date, '%Y%m%d')
        # sex, bdate = dict_sex_bday[pid]
        # age = (study_date - bdate).days/365
        # sexes.append(sex)
        # ages.append(age)
    # display(pd.DataFrame(sexes, columns=['PatientSex']).value_counts())
    # display(pd.DataFrame(ages, columns=['PatientAge']).describe().T)

    df.set_index("filename", inplace=True, drop=True)

    df_flags = df[FLAG_COLS]
    df_flag_stats = pd.DataFrame(df_flags.sum(), columns=["count"])
    display(df_flag_stats.T)

    df_flag_stats["flag"] = df_flag_stats.apply(
        lambda row: "{}\n({})".format(row.name, row["count"]), axis=1)
    # sns.barplot(data=df_flag_stats, x='flag', y='count')
    # plt.show()

    df_true = df[[
        c for c in df.columns
        if (c.startswith("true") and not c.endswith("curve_indices"))
    ]]
    display(HTML(f"<h2>Reference</h2>"))
    display(df_true.describe())
    pred_col_order = [
        "pred " + c.replace("true ", "") for c in df_true.columns
    ]
    # df_pred = df[[c for c in df.columns if c.startswith('pred')]]
    df_pred = df[pred_col_order]
    display(HTML(f"<h2>AI</h2>"))
    display(df_pred.describe())

    df_diff = pd.DataFrame(df_true.values - df_pred.values,
                           columns=[c.replace("true ", "") for c in df_true])
    df_diff.index = df_true.index
    display(HTML(f"<h2>Reference - AI</h2>"))
    desc = df_diff.describe()
    df_diff_dropped = df.dropna()
    new_rows = defaultdict(list)
    for col_true in df_true.columns:
        col_pred = col_true.replace("true", "pred")
        r, p, lo, hi = pearsonr_ci(df_diff_dropped[col_true],
                                   df_diff_dropped[col_pred])
        new_rows["r"].append(r)
        new_rows["R^2"].append(r**2)
        df_cmp = df[[col_true, col_pred]]
        df_cmp = df_cmp.dropna()
        diff = df_cmp[col_true] - df_cmp[col_pred]
        new_rows["RMS"].append(np.sqrt((diff**2).mean()))
        result = stats.linregress(df_cmp[col_true], df_cmp[col_pred])
        new_rows["regression slope"].append(result.slope)
        new_rows["regression intercept"].append(result.intercept)
    for k, v in new_rows.items():
        desc.loc[k] = v
    display(desc)

    display(HTML(f"<h2>|Reference - AI|</h2>"))
    display(df_diff.abs().describe())

    df["op"] = df["術後"].map(lambda e: "post" if e else "pre")
    df["eos"] = df["EOS"].map(lambda e: "EOS" if e else "XR")
    col_type = 'Image Type'
    df[col_type] = df.apply(lambda row: f'{row.eos} & {row.op}', axis=1)
    hue = col_type
    hue_order = ['XR & pre', 'XR & post', 'EOS & pre', 'EOS & post']

    cbs = []
    for col_name in ['Heatmap', 'Points', 'VertebralNumber', 'MajorApex'] + [
            c[5:] for c in df.columns
            if c.startswith('pred ') and not c.endswith('curve_indices')
    ]:
        cbs.append(t_cb.render(name=col_name, label=col_name, checked='checked'))
    cbs = '\n'.join(cbs)
    cbs = '<h2>Show/Hide</h2>' + cbs + '''
        <button onclick="change_all('visible')">Show All</button>
        <button onclick="change_all('hidden')">Hide All</button>
'''

    display(HTML(f"<h2>Plots</h2>"))
    for col in angle_cols:
        display(HTML(f"<h3>{col}</h3>"))
        col_x = "true " + col
        col_y = "pred " + col
        df_sub = df[[col_type, col_x, col_y]].copy()
        df_sub = df_sub.dropna()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        ax = sns.scatterplot(data=df_sub,
                             x=col_x,
                             y=col_y,
                             hue=hue,
                             alpha=min(1, 100 / len(df_sub)))
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.plot(lims, lims, "gray", linestyle="dotted", label="y = x")
        ax.set_xlabel("Reference")
        ax.set_ylabel("AI")
        # r, p = stats.pearsonr(df_dropped[col_x], df_dropped[col_y])
        r, p, lo, hi = pearsonr_ci(df_dropped[col_x], df_dropped[col_y])
        ax.set_title("{} (r={:.03f} [{:.03f} - {:.03f}])".format(
            col, r, lo, hi))
        plt.legend()
        plt.subplot(1, 2, 2)
        df_d = df_diff[[col]].dropna().copy()
        df_d[col_type] = df.loc[df_d.index][col_type]
        sns.histplot(data=df_d, x=col, hue=hue, hue_order=hue_order)
        plt.title("(μ={:.2f}, σ={:.2f}, RMS={:.2f})".format(
            df_d[col].mean(), df_d[col].std(),
            np.sqrt((df_d[col].to_numpy()**2).mean())))
        plt.show()

        axes = sns.lmplot(data=df_sub,
                          x=col_x,
                          y=col_y,
                          col=hue,
                          col_order=hue_order,
                          scatter_kws=(dict(alpha=min(1, 100 / len(df_sub)))))
        for ax in axes.axes_dict.values():  #ax = plt.gca()
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            ax.plot(lims, lims, "gray", linestyle="dotted", label="y = x")
            ax.set_xlabel("Reference")
            ax.set_ylabel("AI")
            # r, p = stats.pearsonr(df_dropped[col_x], df_dropped[col_y])
            # r, p, lo, hi = pearsonr_ci(df_dropped[col_x], df_dropped[col_y])
            # ax.set_title("{} (r={:.03f} [{:.03f} - {:.03f}])".format(col, r, lo, hi))
            plt.legend()
        plt.show()

        # flex_style = "display:flex;flex-direction:row;flex-wrap:wrap;"
        flex_style = "display:flex;flex-direction:row;"
        N = 10
        if col in key_cols:
            # for best_worst, sls in [('Best',slice(N)), ('Median',slice(len(df_d)//2-N//2,int(len(df_d)//2+np.ceil(N/2)))), ('Worst',slice(-N,None))]:
            third_q = (
                "Third quartile",
                slice(3 * (len(df_d) // 4) - N // 2,
                      int(3 * (len(df_d) // 4) + np.ceil(N / 2))),
            )
            p_90 = (
                "90 percentile",
                slice(9 * (len(df_d) // 10) - N // 2,
                      int(9 * (len(df_d) // 10) + np.ceil(N / 2))),
            )
            p_95 = (
                "95 percentile",
                slice(95 * (len(df_d) // 100) - N // 2,
                      int(95 * (len(df_d) // 100) + np.ceil(N / 2))),
            )
            for best_worst, sls in [
                ("Median",
                 slice(
                     len(df_d) // 2 - N // 2,
                     int(len(df_d) // 2 + np.ceil(N / 2)))),
                    # third_q,
                    # p_90,
                    p_95,
                ("Worst", slice(-N, None)),
            ]:
                # for best_worst, sls in [('Median',slice(len(df_d)//2-N//2,int(len(df_d)//2+np.ceil(N/2)))), ('Worst',slice(-N,None))]:
                display(HTML(f"<h4>{best_worst}</h4>"))
                df_reps = df_d[[col]].abs().sort_values(col)[sls]
                display(df_reps)
                if include_visualization:
                    d_vis = {}
                    for subdir, label in [("reference", "Reference"),
                                          ("auto", "AI")]:
                        display(HTML(f"<h5>{label}</h5>"))
                        svgs = []
                        for file_id, row in df_reps.iterrows():
                            if STORED_CURVE and name == 'ap' and subdir == 'auto':
                                svg_filename = result_dir / subdir / 'ap_whole_w_curve_indices' / f"{file_id}.svg"
                            else:
                                svg_filename = result_dir / subdir / f"{name}_whole" / f"{file_id}.svg"
                            if svg_filename.exists():
                                with open(svg_filename) as f:
                                    svgs.append(
                                        '<div class="sample" title="{}">{}</div>'
                                        .format(file_id, f.read()))
                            else:
                                with open(svg_filename.with_suffix(
                                        '.html')) as f:
                                    soup = BeautifulSoup(
                                        f.read(), 'html.parser')
                                svg = str(soup.select_one('svg'))
                                d_vis[(file_id, label)] = svg
                                svgs.append(
                                    '<div class="sample" title="{}">{}</div>'.
                                    format(file_id, svg))
                        display(
                            HTML('<div style="{}">{}</div>'.format(
                                flex_style, "\n".join(svgs))))
                    # Create independent html
                    html = [
                        f'<html><head><meta charset="utf-8">{STYLE}{JAVASCRIPT}</head><body>'
                    ]
                    html.append(f'<div id="control">{cbs}</div>')
                    for file_id, row in df_reps.iterrows():
                        html.append('<div class="patient">')
                        html.append(f'<h2>{file_id}</h2>')
                        html.append('<div class="sbs">')
                        for label in ["Reference", "AI"]:
                            svg = d_vis[(file_id, label)]
                            elm = f'<div class="img"><h3>{label}</h3>{svg}</div>'
                            html.append(elm)
                        html.append('</div></div>')
                    html.append('</body></html>')
                    sbs_dir = result_dir / (
                        'side_by_side' +
                        ('_w_curve_indices' if STORED_CURVE else ''))
                    sbs_dir.mkdir(parents=True, exist_ok=True)
                    with open(sbs_dir / f'{col}_{best_worst}.html', 'w') as f:
                        f.write('\n'.join(html))
    #         break
    # break
