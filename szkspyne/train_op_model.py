#%%
from pathlib import Path
from szkspyne.utils.op import OpData, OpDataLine, OpDataset
from szkspyne.type.op import OpParams, OpParamsLine
import random
import argparse
from logzero import logger
from szkspyne.timm_model import LitTimm
import json

parser = argparse.ArgumentParser(description='Train Operation Model')
parser.add_argument('--data_root', type=str, default='../../data/generative_dataset/op_dataset/',
                    help='Root directory for the operation dataset')
parser.add_argument('--output_root', type=str, default='../../result/generative/op/',
                    help='Root directory for saving the output results')
# max_epochs, batch_size, and other hyperparameters
parser.add_argument('--max_epochs', type=int, default=128, help='Maximum number of epochs for training')

LitTimm.add_model_specific_args(parser)
args = parser.parse_args()
data_root = Path(args.data_root)
output_root = Path(args.output_root)
output_root.mkdir(exist_ok=True, parents=True)

with open(output_root / 'args.json', 'w') as f:
    json.dump(vars(args), f, indent=2)

logger.info(f'Load data from {data_root}')
train_dir = data_root / 'train'
train_val_data_list = OpData.from_directory(train_dir)
# split train and validation datasets with 90% for training and 10% for validation
random.seed(42)  # For reproducibility
random.shuffle(train_val_data_list)
split_idx = int(len(train_val_data_list) * 0.9)
from torchvision.transforms import ToTensor
from PIL import Image
IMAGE_SIZE = (256, 512)  # width, height

# image_transform = ToTensor()
def image_transform(image: Image):
    """Convert PIL image to tensor and normalize."""
    image = image.resize(IMAGE_SIZE)
    image = ToTensor()(image)
    return image

vector_transform = lambda x: torch.tensor(x, dtype=torch.float32)
dataset_kwargs = {'image_transform': image_transform, 'vector_transform': vector_transform}
train_dataset = OpDataset(train_val_data_list[:split_idx], **dataset_kwargs)
val_dataset = OpDataset(train_val_data_list[split_idx:], **dataset_kwargs)
vector_length = len(train_val_data_list[0].to_vector())

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
# %%
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

import torch

# Define model parameters
model_name = args.model_name
in_chs = 1
out_chs = vector_length
hparams = {'lr': 1e-3}
logger.info(f'Creating model {model_name} with input channels {in_chs} and output channels {out_chs}')
input_shape_hw = (IMAGE_SIZE[1], IMAGE_SIZE[0])  # height, width
model = LitTimm(model_name=model_name, in_chs=in_chs, out_chs=out_chs, input_shape_hw=input_shape_hw, hparams=hparams)
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
# Define callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_root / 'checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=8,
    mode='min',
    verbose=True
)

# Create the Trainer
trainer = Trainer(
    max_epochs=128,
    callbacks=[checkpoint_callback, early_stopping_callback],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision=16 if torch.cuda.is_available() else 32,
    devices=1 if torch.cuda.is_available() else None
)
# Train the model
logger.info('Starting training...')
trainer.fit(model, train_loader, val_loader)
# Save the model
logger.info('Saving the model...')
model_path = output_root / f'models/{model_name}_op_model.pth'
model_path.parent.mkdir(exist_ok=True, parents=True)  # Ensure the directory exists
# load the best weight from checkpoint
if checkpoint_callback.best_model_path:
    logger.info(f"Loading best model from {checkpoint_callback.best_model_path}")
    # model.model.load_state_dict(torch.load(checkpoint_callback.best_model_path, map_location='cpu'))
    model = LitTimm.load_from_checkpoint(checkpoint_callback.best_model_path)

else:
    logger.warning("No best model found, saving the current model state.")
model.model.to('cpu')  # Move model to CPU before saving
torch.save(model.model.state_dict(), model_path)
print(f"Model saved to {model_path}")

test_dir = data_root / 'test'
test_data_list = OpData.from_directory(test_dir)
test_dataset = OpDataset(test_data_list, **dataset_kwargs, image_only=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
# Evaluate the model on the test set
logger.info('Evaluating the model on the test set...')
output = trainer.predict(model, test_loader)

# Process the output
predictions = []
for batch in output:
    for pred in batch:
        predictions.append(pred.numpy())

pred_params = [OpParams.from_vector(pred) for pred in predictions]
filenames = [Path(data.image_path).with_suffix('.json').name for data in test_data_list]
op_data_lines = [OpDataLine(filename=fn, content=OpData(op_params=op_params, image_path=data.image_path)) 
                for fn, op_params, data in zip(filenames, pred_params, test_data_list)]
# Save the predictions to JSON files
OpDataLine.save_to_json(op_data_lines, output_root / 'predictions.ndjson')