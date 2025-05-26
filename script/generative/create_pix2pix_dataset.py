#%%
from pathlib import Path
import json

data_dir = Path('../../data/paired_dataset')
image_dir = data_dir / 'images'
output_dir = data_dir / 'pix2pix'

with open(data_dir / 'paired_dataset.ndjson', 'r') as f:
    lines = f.readlines()
    pre_post_pairs = [json.loads(line) for line in lines]
print(len(pre_post_pairs))
# %%
import random
random.shuffle(pre_post_pairs)
val_ratio = 0.1
n_val = int(len(pre_post_pairs) * val_ratio)
n_train = len(pre_post_pairs) - n_val
print('train', n_train, 'val', n_val)
train_pairs = pre_post_pairs[:n_train]
val_pairs = pre_post_pairs[n_train:]
test_pairs = val_pairs
print('train', len(train_pairs), 'val', len(val_pairs))

#%%
from PIL import Image
from tqdm import tqdm
image_size = (256, 512)

for name, pairs in [
    ('train', train_pairs),
    ('val', val_pairs),
    ('test', test_pairs),
]:
    print(name, len(pairs))
    outdir = output_dir / name
    outdir.mkdir(exist_ok=True, parents=True)
    for i, pair in enumerate(tqdm(pairs)):
        pre_op = pair['pre_op']
        post_op = pair['post_op']
        pre_op_path = image_dir / pre_op['filename']
        post_op_path = image_dir / post_op['filename']
        pre_op_image = Image.open(pre_op_path)
        post_op_image = Image.open(post_op_path)
        pre_op_image = pre_op_image.resize(image_size)
        post_op_image = post_op_image.resize(image_size)
        # horizontal concatenation
        pair_image = Image.new('L', (image_size[0] * 2, image_size[1]))
        pair_image.paste(pre_op_image, (0, 0))
        pair_image.paste(post_op_image, (image_size[0], 0))
        pair_image.save(outdir / f'{i:03d}.jpg')
        # break
# %%
