import timm
import pytorch_lightning as pl
import torch

class LitTimm(pl.LightningModule):
    def __init__(self, model_name, in_chs, out_chs, input_shape_hw, hparams):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, in_chans=in_chs,num_classes=0, global_pool='')
        feature_shape = self.calculate_feature_shape(model_name, input_shape_hw[0], input_shape_hw[1])
        int_feature_size = 128   
        # conv block
        self.reduce_feature = torch.nn.Sequential(
            torch.nn.Conv2d(feature_shape[0], 2*int_feature_size, kernel_size=3, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(2*int_feature_size),
            torch.nn.Conv2d(2*int_feature_size, int_feature_size, kernel_size=3, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(int_feature_size),
        )
        self.flatten = torch.nn.Flatten()
        feature_dim = int_feature_size * (feature_shape[1]-4) * (feature_shape[2]-4)  # Adjusted for pooling and convolution
        self.regresasor = torch.nn.Linear(feature_dim, out_chs)
        self.model = torch.nn.Sequential(self.encoder, self.reduce_feature, self.flatten, self.regresasor)
        self.loss = torch.nn.MSELoss()
        self.lr = hparams['lr']
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitTimm")
        parser.add_argument('--lr', help='Learning rate: %(default)s', metavar='<value>', type=float, default=1e-3)
        parser.add_argument(
            '--model_name', help='Model name: %(default)s', metavar='<name>', default='efficientnet-b5'
        )
        
        return parent_parser
    
    @staticmethod
    def calculate_feature_shape(model_name, width, height):
        model = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=0, global_pool='')
        with torch.no_grad():
            x = torch.zeros(1, 1, height, width)
            features = model(x)
        return features.shape[1:]  # Return the shape of the last feature map

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test LitTimm model')
    parser.add_argument('--model_name', type=str, default='efficientnet_b5', help='Model name')
    parser.add_argument('--in_chs', type=int, default=1, help='Input channels')
    parser.add_argument('--out_chs', type=int, default=100, help='Output channels')
    # image size
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    
    args = parser.parse_args()
    
    input_shape_hw = (args.height, args.width)  # height, width
    model = LitTimm(model_name=args.model_name, in_chs=args.in_chs, out_chs=args.out_chs, input_shape_hw=input_shape_hw, hparams={'lr': 0.01})
    from torchinfo import summary
    shape = (1, args.in_chs, args.height, args.width)  # batch size, channels, height, width
    summary(model, shape, device='cpu')
    # print(LitTimm.calculate_feature_shape(args.model_name, args.width, args.height))