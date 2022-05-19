from ast import ListComp
import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.optim as optim

import pytorch_lightning as pl


class LitSMP(pl.LightningModule):
    def __init__(self, in_chs, out_chs, hparams):
        super().__init__()
        kwargs = dict(encoder_name=hparams.encoder_name,
                      encoder_weights=hparams.encoder_weights,
                      activation=None,
                      in_channels=in_chs,
                      classes=out_chs)
        model = smp.DeepLabV3Plus(**kwargs)
        self.model = model
        self.loss = torch.nn.MSELoss()
        self.lr = hparams.lr
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitSMP")
        parser.add_argument('--lr',
                            help='Learning rate: %(default)s',
                            metavar='<value>',
                            type=float,
                            default=1e-3)
        parser.add_argument('--encoder_name',
                            help='encoder name: %(default)s',
                            metavar='<name>',
                            default='efficientnet-b4')
        parser.add_argument('--encoder_weights',
                            help='encoder weights: %(default)s',
                            metavar='<name>',
                            default='imagenet')
        return parent_parser

    def forward(self, x):
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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return torch.clamp(torch.round(255 * self(batch)), 0,
                           255).to(torch.uint8)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=.1,
                                                         patience=4)
        scheduler = {'scheduler': scheduler, 'monitor': 'train_loss', 'interval': 'epoch'}
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            # 'monitor': 'train_loss'
        }

    def summary(self, shape):
        from torchinfo import summary
        return summary(self.model, shape, device='cpu')


class Seg2Cls(nn.Module):
    def __init__(self, base_model, aux_params):
        super(Seg2Cls, self).__init__()
        self.encoder = base_model.encoder
        # self.classification_head = smp.base.ClassificationHead(
        #     in_channels=self.encoder.out_channels[-1], **aux_params)
        middle_channels = 64
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(2),
                nn.Conv2d(self.encoder.out_channels[-1], middle_channels, kernel_size=3),
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
                nn.BatchNorm2d(middle_channels),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(middle_channels, 1, bias=True)
            )

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x)[-1]
        labels = self.classification_head(features)
        return labels

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def save_classifier(self, filename):
        torch.save(self.classification_head.state_dict(), filename)

    def load_classifier(self, filename):
        self.classification_head.load_state_dict(torch.load(filename))

    # def summary(self, shape):
    #     from torchinfo import summary
    #     return summary(self, shape, device='cpu')

class LitCls(pl.LightningModule):
    def __init__(self, seg_model, aux_params, hparams):
        super().__init__()
        self.model = Seg2Cls(seg_model, aux_params)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.lr = hparams.lr
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitCls")
        return parent_parser

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.classification_head.parameters(), self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=.1,
                                                         patience=4)
        return {
            'optimizer': optimizer,
            # 'scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def summary(self, shape):
        from torchinfo import summary
        return summary(self.model, shape, device='cpu')

def show_summary(args):
    model = LitSMP(1, 6, args)
    print(model.summary((1, 1, 512, 512)))

def create_onnx(args):
    model = LitSMP(args.in_chs, args.out_chs, args).model
    def set_align_false(model):
        for l in model.children():
            set_align_false(l)
        if hasattr(model, 'align_corners'):
            print('Set align corner false')
            print(model, model.align_corners)
            setattr(model, 'align_corners', False)
            # model.align_corners = False
    set_align_false(model)

    model.load_state_dict(torch.load(args.weights))
    set_align_false(model)
    try:
        model.encoder.set_swish(memory_efficient=False)
    except Exception as e:
        print(e)
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn((1, 1, 768, 768), requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         args.output,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=12,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size', 3: 'image_width'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size', 3: 'image_width'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Pytorch lightning with segmentation_models_pytorch.')
    LitSMP.add_model_specific_args(parser)
    subparsers = parser.add_subparsers(help='Commands')
    parser_summary = subparsers.add_parser('summary', help="Show the network summary of the specified model.")
    parser_summary.set_defaults(func=show_summary)

    parser_onnx = subparsers.add_parser('onnx', help="Create ONNX model.")
    parser_onnx.set_defaults(func=create_onnx)
    parser_onnx.add_argument('in_chs',
                    help='The # of input channels',
                    type=int)
    parser_onnx.add_argument('out_chs',
                    help='The # of output channels',
                    type=int)
    parser_onnx.add_argument('weights',
                    help='Path of the trained weight')
    parser_onnx.add_argument('output',
                    help='Output onnx filename')
    args = parser.parse_args()

    has_func = hasattr(args, 'func')
    if not has_func:
        import sys
        parser.error('too few arguments')
        sys.exit(1)
    args.func(args)
