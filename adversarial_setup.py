import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from models import MobileGenerator, Discriminator, AgeClassifier, FeatureExtractor
from dataloader import DataLoaderAge, DataLoaderGAN


class AgeClassifier(pl.LightningModule):
    def __init__(self, image_dir, text_dir, image_size, batch_size):
        super(AgeClassifier, self).__init__()
        self.model = AgeClassifier()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = image_size
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-2)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(DataLoaderAge(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        transforms=None,
                                        is_train=True),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(DataLoaderAge(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        transforms=None,
                                        is_train=False),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=False)


class GenAdvNet(pl.LightningModule):
    def __init__(self, image_dir, text_dir, image_size, batch_size):
        super(GenAdvNet, self).__init__()
        self.generator = MobileGenerator(num_blocks=6)
        self.discriminator = Discriminator()
        self.classifier = AgeClassifier()
        self.features = FeatureExtractor()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = image_size
        self.batch_size = batch_size

    def forward(self, x):
        return self.generator(x)

    def train_step(self, batch, batch_idx):
        # Will write later
        pass

    def val_step(self, batch, batch_idx):
        # Will write later
        pass

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        transforms=None,
                                        is_train=True),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        transforms=None,
                                        is_train=False),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=False)
