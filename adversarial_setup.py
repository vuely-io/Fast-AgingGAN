import os
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from dataloader import DataLoaderAge, DataLoaderGAN
from models import ResnetGenerator, Discriminator, AgeClassifier


class AgeModule(pl.LightningModule):
    def __init__(self, image_dir, text_dir, image_size, batch_size):
        super(AgeModule, self).__init__()
        self.model = AgeClassifier()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size

    def forward(self, x):
        x, features = self.model(x)
        return x, features

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(DataLoaderAge(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        is_train=True),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(DataLoaderAge(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        is_train=False),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=False,
                          drop_last=True)


class GenAdvNet(pl.LightningModule):
    def __init__(self, image_dir, text_dir, image_size, batch_size):
        super(GenAdvNet, self).__init__()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size

        self.generator = ResnetGenerator(4, 3, 64, norm_layer=torch.nn.BatchNorm2d, use_dropout=False, n_blocks=9)
        self.discriminator = Discriminator(3, )
        self.classifier = AgeClassifier()

        # TODO find a nicer way to do this:
        ckpt_dir = './lightning_logs/version_0/checkpoints/'
        ckpt = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
        ckpt = torch.load(ckpt)
        new_state_dict = {}
        for k, v in ckpt['state_dict'].items():
            # Remove the mode.model repetition from the key name
            new_state_dict[k[6:]] = v

        self.classifier.load_state_dict(new_state_dict)
        for p in self.classifier.parameters():
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
            d1_real_loss = torch.pow(d1_logit - 1.0, 2).mean()
            d2_fake_loss = torch.pow(d2_logit, 2).mean()
            d3_fake_loss = torch.pow(d3_logit, 2).mean()

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

            # Get age prediction
            gen_age, gen_features = self.classifier(self.aged_image)
            _, src_features = self.classifier(batch['src_image_cond'][:, :3, ...])

            # Get generator losses
            d3_real_loss = 0.5 * torch.pow(d3_logit - 1.0, 2).mean() * 75
            age_loss = self.criterion_ce(gen_age, batch['true_label']) * 30
            feature_loss = self.criterion_mse(gen_features, src_features) * 5e-5

            g_loss = (d3_real_loss + age_loss + feature_loss) / 75.0

            # log sampled images
            if batch_idx % 200 == 0:
                grid = torchvision.utils.make_grid(batch['src_image_cond'][:6, :3, ...],
                                                   normalize=True,
                                                   range=(0, 1),
                                                   scale_each=True)
                self.logger.experiment.add_image('source_image', grid, 0)
                grid = torchvision.utils.make_grid(self.aged_image[:6],
                                                   normalize=True,
                                                   range=(0, 1),
                                                   scale_each=True)
                self.logger.experiment.add_image('generated_images', grid, 0)
                grid = torchvision.utils.make_grid(batch['true_image'][:6],
                                                   normalize=True,
                                                   range=(0, 1),
                                                   scale_each=True)
                self.logger.experiment.add_image('target_images', grid, 0)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def on_epoch_end(self):
        if not os.path.exists('model_weights'):
            os.makedirs('model_weights')
        torch.save(self.generator.state_dict(), 'model_weights/gen.pth')
        torch.save(self.discriminator.state_dict(), 'model_weights/disc.pth')

    def configure_optimizers(self):
        return [torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6),
                torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)], []

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        is_train=True),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=True,
                          drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                        text_dir=self.text_dir,
                                        image_size=self.image_size,
                                        is_train=False),
                          batch_size=self.batch_size,
                          num_workers=4,
                          shuffle=False,
                          drop_last=True)
