import os
from tensorboardX import SummaryWriter

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


class GenAdvNet(object):
    def __init__(self, image_dir, text_dir, image_size, batch_size):
        super(GenAdvNet, self).__init__()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size

        self.generator = ResnetGenerator(8, 3, 64, norm_layer=torch.nn.BatchNorm2d, use_dropout=False, n_blocks=9)
        self.discriminator = Discriminator(3)
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
        self.classifier.eval()

        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()

        self.d_optim = torch.optim.Adam(params=self.discriminator.parameters(), lr=1e-4)
        self.g_optim = torch.optim.Adam(params=self.generator.parameters(), lr=1e-4)

        self.writer = SummaryWriter()

    def fit(self):
        train_queue = DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                               text_dir=self.text_dir,
                                               batch_size=self.batch_size),
                                 batch_size=self.batch_size,
                                 num_workers=4,
                                 shuffle=True,
                                 drop_last=True)

        for step, batch in enumerate(train_queue):
            source_img_128, true_label_img, true_label_128, true_label_64, fake_label_64, true_label = batch
            # Train discriminator
            self.discriminator.zero_grad()
            # Obtain aged image from generator
            self.aged_image = self.generator(torch.cat([source_img_128, true_label_128], dim=1))
            # Calculate loss on all real batch
            d1_logit = self.discriminator(true_label_img, true_label_64)
            d2_logit = self.discriminator(true_label_img, fake_label_64)
            d3_logit = self.discriminator(self.aged_image.detach(), true_label_64)

            # Do label smoothing and create targets:
            b, c, h, w = d1_logit.shape
            valid_target = torch.ones(d1_logit.shape) - torch.empty(b, c, h, w).uniform_(0, 0.1)
            fake_target = torch.empty(b, c, h, w).uniform_(0.0, 0.1)

            # Calculate all real loss
            d1_real_loss = self.criterion_bce(d1_logit, valid_target.cuda())
            # Calculate real image, fake condition loss
            d2_fake_loss = self.criterion_bce(d2_logit, fake_target.cuda())
            # Calculate fake image, real condition loss
            d3_fake_loss = self.criterion_bce(d3_logit, fake_target.cuda())
            # Calculate the average loss
            d_loss = 0.5 * (d1_real_loss + 0.5 * (d2_fake_loss + d3_fake_loss))
            # Calculate gradients wrt all losses
            d_loss.backward()
            # Apply the gradient update
            self.d_optim.step()

            # Train the generator
            self.generator.zero_grad()
            # Get age prediction
            gen_age, gen_features = self.classifier(self.aged_image)
            _, src_features = self.classifier(source_img_128)
            # Get adversarial loss
            d3_logit = self.discriminator(self.aged_image, true_label_64)
            # Label Smoothing
            b, c, h, w = d3_logit.shape
            valid_target = torch.ones(d3_logit.shape) - torch.empty(b, c, h, w).uniform_(0, 0.1)
            # Get losses
            d3_real_loss = self.criterion_bce(d3_logit, valid_target.cuda())
            age_loss = self.criterion_ce(gen_age, true_label)
            feature_loss = self.criterion_mse(gen_features.view(b, -1), src_features.view(b, -1))
            # Get avg loss
            g_loss = (d3_real_loss + age_loss + feature_loss)
            # Calculate gradients
            g_loss.backward()
            # Apply update to the generator
            self.g_optim.step()
            # log sampled images
            if step % 200 == 0:
                self.writer.add_scalar('g_loss', g_loss.item())
                self.writer.add_image('d_loss', d_loss.item())
                grid = torchvision.utils.make_grid(source_img_128,
                                                   normalize=True,
                                                   range=(0, 1),
                                                   scale_each=True)
                self.writer.add_image('source_image', grid, step)
                grid = torchvision.utils.make_grid(self.aged_image,
                                                   normalize=True,
                                                   range=(0, 1),
                                                   scale_each=True)
                self.writer.add_image('generated_images', grid, step)
                grid = torchvision.utils.make_grid(true_label_img,
                                                   normalize=True,
                                                   range=(0, 1),
                                                   scale_each=True)
                self.writer.add_image('target_images', grid, step)
                self.writer.flush()
                # Save the weights
                torch.save(self.generator.state_dict(), 'models/gen.pth')
                torch.save(self.discriminator.state_dict(), 'models/disc.pth')
