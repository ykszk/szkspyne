#%%
from pathlib import Path
from szkspyne.type.op import OpParams, OpParamsLine
from szkspyne.utils.op import OpData, OpDataLine, OpDataset
import pandas as pd
import os
pd.options.display.max_columns = 2000 # show all columns in DataFrame

result_filename = os.environ.get('RESULT_FILENAME', '../../result/generative/op/predictions.ndjson')
pair_filename = os.environ.get('PAIR_FILENAME', '../../data/generative_dataset/op_dataset/pre_post_pairs.csv')
ref_data_dir = os.environ.get('REF_DATA_DIR', '../../data/generative_dataset/op_dataset/test')

result_ndjson = OpDataLine.from_ndjson(result_filename)
pred_op_data = [line.content for line in result_ndjson]
pred_dict = {Path(line.filename).stem: line.content for line in result_ndjson}
#%%
df_pairs = pd.read_csv(pair_filename)
df_pairs
pre_post_dict = {Path(row['pre_op']).stem: Path(row['post_op']).stem for _, row in df_pairs.iterrows()}
post_pre_dict = {v: k for k, v in pre_post_dict.items()}
pair_dict = pre_post_dict | post_pre_dict

# %%
ref_op_data = OpData.from_directory(Path(ref_data_dir))
ref_dict = {Path(op_data.image_path).stem: op_data for op_data in ref_op_data}


# %%
import matplotlib.pyplot as plt
from IPython.display import display

for pred in pred_op_data:
    print(f'Processing {pred.image_path}')
    ref = ref_dict[Path(pred.image_path).stem]
    pred_params, ref_params = pred.op_params, ref.op_params
    df = pd.DataFrame([pred_params.to_vector(), ref_params.to_vector()])
    df.index = ['Predicted', 'Reference']
    columns = ['Left ' + label for label in pred.op_params.LABELS] + \
             ['Right ' + label for label in pred.op_params.LABELS] + \
             ['Tilt Angle ' + str(i + 1) for i in range(len(pred.op_params.tilt_angles))]
    df.columns = columns
    display(df.iloc[:, :2 * len(pred.op_params.LABELS)])
    display(df.iloc[:, 2 * len(pred.op_params.LABELS):])
    image = pred.load_image()
    pre_post_id = pair_dict[Path(pred.image_path).stem]
    pre_post_ref = ref_dict[pre_post_id]
    pre_post_image = pre_post_ref.load_image()
    # plt.imshow(image, 'gray')
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image, 'gray')
    ax1.set_title('Input')
    ax2.imshow(pre_post_image, 'gray')
    ax2.set_title('Pre/Post')
    plt.show()

