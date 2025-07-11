from typing import List
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from szkmipy import boundingbox as bb
import img_utils
import numpy as np
import torch.nn.functional as F

DOWN_FACTOR = 256
bb.enable_torch()


class LitSMP(pl.LightningModule):
    def __init__(self, in_chs, out_chs, hparams):
        super().__init__()
        kwargs = dict(
            encoder_name=hparams.encoder_name,
            encoder_weights=hparams.encoder_weights,
            activation=None,
            in_channels=in_chs,
            classes=out_chs,
        )
        model = smp.DeepLabV3Plus(**kwargs)
        self.model = model
        # self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.lr = hparams.lr
        self.auto_crop_enabled = False
        self.auto_crop_height = 1024
        self.auto_crop_thresh = 0.1
        self.auto_crop_margin = 0.1
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitSMP")
        parser.add_argument('--lr', help='Learning rate: %(default)s', metavar='<value>', type=float, default=1e-3)
        parser.add_argument(
            '--encoder_name', help='encoder name: %(default)s', metavar='<name>', default='efficientnet-b4'
        )
        parser.add_argument(
            '--encoder_weights', help='encoder weights: %(default)s', metavar='<name>', default='imagenet'
        )
        return parent_parser

    def forward(self, x):
        if self.auto_crop_enabled:
            return self.auto_crop_apply(x, self.auto_crop_height, self.auto_crop_thresh, self.auto_crop_margin)
        else:
            return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log(
            'train_loss', loss, sync_dist=True
        )  # sync_dist=True https://github.com/PyTorchLightning/pytorch-lightning/issues/5604#issuecomment-785314359
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return torch.sigmoid(self(batch))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
        scheduler = {'scheduler': scheduler, 'monitor': 'train_loss', 'interval': 'epoch'}
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            # 'monitor': 'train_loss'
        }

    def summary(self, shape):
        from torchinfo import summary

        return summary(self.model, shape, device='cpu')

    def enable_auto_crop(self, height, crop_thresh=0.1, margin=0.1):
        self.auto_crop_enabled = True
        self.auto_crop_height = height
        self.auto_crop_thresh = crop_thresh
        self.auto_crop_margin = margin

    def auto_crop_apply(self, xs, height, crop_thresh=0.1, margin=0.1) -> torch.Tensor:
        heatmaps = self.model(xs)
        # remove channel dim
        heatmaps = torch.sigmoid(torch.max(heatmaps, dim=1)[0])
        resized_xs = []
        cropped_shapes = []
        bboxes = []
        for heatmap, x in zip(heatmaps, xs):
            x = x.detach().cpu().numpy()
            heatmap = heatmap.detach().cpu().numpy()
            bbox = bb.bbox(heatmap > crop_thresh)
            bbox_shape = bbox[1] - bbox[0]
            margin_ratio = 0.1
            margin = (bbox_shape * margin_ratio).astype(int)
            bbox = bb.add_margin_to_bbox(heatmap, bbox, margin)
            # add channel dim to bbox
            bmin, bmax = bbox
            bmin = np.concatenate([np.array([0]), bmin])
            bmax = np.concatenate([np.array([x.shape[0]]), bmax])
            bbox = (bmin, bmax)
            bboxes.append(bbox)
            cropped_x = bb.crop(x, bbox)
            cropped_shapes.append(cropped_x.shape)
            # resized_x = img_utils.resize_image_height(height, cropped_x)
            _scale, width = img_utils.calc_resize_height(height, cropped_x.shape[1], cropped_x.shape[2])
            input_tensor = torch.Tensor(cropped_x[np.newaxis])
            resized_x = F.interpolate(input_tensor, size=(height, width), mode='bilinear')
            resized_xs.append(resized_x[0, 0])  # TODO: handle channel dimension (second dim) properly
        constant_value = int(torch.min(heatmaps))
        del heatmaps
        cropped_batch = torch.Tensor(np.stack(cropped_batch)).to(self.device)

        heatmaps = self.model(cropped_batch)

        # restore original shape
        restored_xs = []
        for heatmap, bbox, cropped_shape in zip(heatmaps, bboxes, cropped_shapes):
            # heatmap = heatmap.detach().cpu().numpy()
            # heatmap = img_utils.resize_image_height(cropped_shape[1], heatmap)
            _scale, width = img_utils.calc_resize_height(cropped_shape[1], heatmap.shape[1], heatmap.shape[2])
            heatmap = F.interpolate(heatmap[torch.newaxis], size=(cropped_shape[1], width), mode='bilinear')
            heatmap = heatmap[0, :, 0 : cropped_shape[1], 0 : cropped_shape[2]]
            # heatmap = bb.uncrop(heatmap, xs[0].shape, bbox)
            # print(heatmap.shape)
            # remove channel dim from bbox
            bbox = (bbox[0][1:], bbox[1][1:])
            restored = []
            for i in range(len(heatmap)):
                restored.append(bb.uncrop(heatmap[i], xs[0, 0].shape, bbox, constant_values=constant_value))
            restored = torch.stack(restored)
            restored_xs.append(restored)
        return torch.stack(restored_xs)


def show_summary(args):
    model = LitSMP(1, 6, args)
    print(model.summary((1, 1, 512, 512)))


def create_onnx(args):

    def set_align_false(model):
        for l in model.children():
            set_align_false(l)
        if hasattr(model, 'align_corners'):
            print('Set align corner false')
            print(model, model.align_corners)
            setattr(model, 'align_corners', False)
            # model.align_corners = False

    if args.weights.endswith('.ckpt'):
        model = LitSMP.load_from_checkpoint(args.weights).to('cpu')
        set_align_false(model)
    else:
        model = LitSMP(args.in_chs, args.out_chs, args).model
        model.load_state_dict(torch.load(args.weights))
        set_align_false(model)
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn((1, 1, args.height, 768), requires_grad=True)

    # Export the model
    with torch.amp.autocast(device_type='cpu', enabled=args.amp):
        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            args.output,  # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file
            # opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['modelInput'],  # the model's input names
            output_names=['modelOutput'],  # the model's output names
            dynamic_axes={
                'modelInput': {0: 'batch_size', 3: 'image_width'},  # variable length axes
                'modelOutput': {0: 'batch_size', 3: 'image_width'},
            },
            # dynamo=True,
        )
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pytorch lightning with segmentation_models_pytorch.')
    LitSMP.add_model_specific_args(parser)
    subparsers = parser.add_subparsers(help='Commands')
    parser_summary = subparsers.add_parser('summary', help="Show the network summary of the specified model.")
    parser_summary.set_defaults(func=show_summary)

    parser_onnx = subparsers.add_parser('onnx', help="Create ONNX model.")
    parser_onnx.set_defaults(func=create_onnx)
    parser_onnx.add_argument('in_chs', help='The # of input channels', type=int)
    parser_onnx.add_argument('out_chs', help='The # of output channels', type=int)
    parser_onnx.add_argument('weights', help='Path of the trained weight')
    parser_onnx.add_argument('output', help='Output onnx filename')
    parser_onnx.add_argument('--amp', help='Use AMP', action='store_true')
    # height
    parser_onnx.add_argument(
        '--height', help='Height of the input image', type=int, default=768, metavar='<value>'
    )
    args = parser.parse_args()

    has_func = hasattr(args, 'func')
    if not has_func:
        import sys

        parser.error('too few arguments')
        sys.exit(1)
    args.func(args)
