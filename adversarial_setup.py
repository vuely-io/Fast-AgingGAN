from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from dataloader import DataLoaderAge, DataLoaderGAN
from models import MobileGenerator, Discriminator, AgeClassifier, FeatureExtractor


class AgeModule(pl.LightningModule):
    def __init__(self, image_dir, text_dir, image_size, batch_size):
        super(AgeModule, self).__init__()
        self.model = AgeClassifier()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = (image_size, image_size)
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
                                        is_train=True),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(DataLoaderAge(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        is_train=False),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=False)


class GenAdvNet(pl.LightningModule):
    def __init__(self, image_dir, text_dir, image_size, batch_size):
        super(GenAdvNet, self).__init__()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size

        self.generator = MobileGenerator(num_blocks=6)
        self.discriminator = Discriminator()
        self.classifier = AgeClassifier()
        self.features = FeatureExtractor()
        for p in self.classifier.parameters():
            p.requires_grad = False
        for p in self.features.parameters():
            p.requires_grad = False

        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Will write later
        self.aged_image = self.forward(batch['src_image_cond'])
        # Train discriminator
        if optimizer_idx == 0:
            # Get logits from discriminator model
            d1_logit = self.discriminator(batch['true_image'], batch['true_cond'])
            d2_logit = self.discriminator(batch['true_image'], batch['false_cond'])
            d3_logit = self.discriminator(self.aged_image.detach(), batch['true_cond'])

            # Calculate losses
            d1_real_loss = self.criterion_mse(d1_logit, torch.ones(d1_logit.shape))
            d2_fake_loss = self.criterion_mse(d2_logit, torch.zeros(d2_logit.shape))
            d3_fake_loss = self.criterion_mse(d3_logit, torch.zeros(d3_logit.shape))

            d_loss = 1. / 2 * (d1_real_loss + 1. / 2 * (d2_fake_loss + d3_fake_loss))
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # Train generator
        if optimizer_idx == 1:
            d3_logit = self.discriminator(self.aged_image, batch['true_cond'])

            # Extract features
            gen_features = self.features(self.aged_image)
            real_features = self.features(batch['true_image'])

            # Get age prediction
            gen_age = self.classifier(self.aged_image)

            # Get generator losses
            d3_real_loss = self.criterion_mse(d3_logit, torch.ones(d3_logit.shape))
            age_loss = self.criterion_ce(gen_age, batch['true_label']) * 1e-3
            feature_loss = self.criterion_mse(gen_features, real_features) * 1e-5

            g_loss = d3_real_loss + age_loss + feature_loss

            # log sampled images
            if batch_idx % 200 == 0:
                sample_imgs = self.aged_image[:6]
                grid = torchvision.utils.make_grid(sample_imgs, normalize=True, range=(0, 1), scale_each=True)
                self.logger.experiment.add_image('generated_images', grid, 0)
                grid = torchvision.utils.make_grid(batch['true_image'], normalize=True, range=(0, 1), scale_each=True)
                self.logger.experiment.add_image('target_images', grid, 0)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        return [torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-3),
                torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-3)], []

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        is_train=True),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        is_train=False),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=False)
