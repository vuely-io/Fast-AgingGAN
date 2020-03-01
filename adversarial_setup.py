import torch
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataloader import DataLoaderAge, DataLoaderGAN
from models import MobileGenerator, Discriminator, AgeClassifier


class AgeModule(object):
    def __init__(self, image_dir, text_dir, image_size, batch_size, epochs=30):
        self.model = AgeClassifier()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AgeClassifier().to(self.device)
        self.writer = SummaryWriter()

    def fit(self):
        train_queue = DataLoader(DataLoaderAge(image_dir=self.image_dir,
                                               text_dir=self.text_dir,
                                               image_size=self.image_size,
                                               is_train=True),
                                 batch_size=self.batch_size,
                                 num_workers=4,
                                 shuffle=True,
                                 drop_last=True)

        val_queue = DataLoader(DataLoaderAge(image_dir=self.image_dir,
                                             text_dir=self.text_dir,
                                             image_size=self.image_size,
                                             is_train=False),
                               batch_size=self.batch_size,
                               num_workers=4,
                               shuffle=False,
                               drop_last=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        counter = 0
        best_loss = float('inf')
        for epoch in range(self.epochs):
            for step, (x, y) in enumerate(train_queue):
                x, y = x.to(self.device), y.to(self.device)
                self.model.zero_grad()
                y_hat, _ = self.model(x)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                opt.step()

                if step % 50 == 0:
                    print('train_loss', loss.item())
                    self.writer.add_scalar('train_loss', loss.item(), len(train_queue) * epoch + step)
                    self.writer.flush()

            total_steps, total_loss, total_accuracy = 0, 0, 0
            with torch.no_grad():
                for step, (x, y) in enumerate(val_queue):
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat, _ = self.model(x)
                    total_loss += F.cross_entropy(y_hat, y).item()
                    total_accuracy += (torch.argmax(y_hat, dim=1) == y).sum().item()
                    total_steps += 1

            total_loss = total_loss / total_steps
            total_accuracy = total_accuracy / (self.batch_size * total_steps)
            print('val_loss', total_loss, 'val_acc', total_accuracy)
            self.writer.add_scalar('val_loss', total_loss, epoch)
            self.writer.add_scalar('val_acc', total_accuracy, epoch)
            self.writer.flush()

            if total_loss < best_loss:
                best_loss = total_loss
                counter = 0
                # Save weights after every epoch
                torch.save(self.model.state_dict(), 'models/classifier_best.pth')

            if counter == 7:
                break
            counter += 1


class GenAdvNet(object):
    def __init__(self, image_dir, text_dir, image_size, batch_size, epochs=50):
        super(GenAdvNet, self).__init__()
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = MobileGenerator(6).to(self.device)
        self.discriminator = Discriminator(3).to(self.device)
        self.classifier = AgeClassifier().to(self.device)

        self.classifier.load_state_dict(torch.load('models/classifier_best.pth'))
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()

        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()

        self.d_optim = torch.optim.Adam(params=self.discriminator.parameters(), lr=1e-4)
        self.g_optim = torch.optim.Adam(params=self.generator.parameters(), lr=1e-4)

        self.w_gan_loss = 70
        self.w_age_loss = 20
        self.w_feat_loss = 5e-5

        self.writer = SummaryWriter()

    def fit(self):
        train_queue = DataLoader(DataLoaderGAN(image_dir=self.image_dir,
                                               text_dir=self.text_dir,
                                               batch_size=self.batch_size),
                                 batch_size=self.batch_size,
                                 num_workers=4,
                                 shuffle=True,
                                 drop_last=True)

        for epoch in range(self.epochs):
            for step, batch in enumerate(train_queue):
                batch = [x.to(self.device) for x in batch]
                source_img_128, true_label_img, true_label_128, true_label_64, fake_label_64, true_label = batch
                # Train discriminator
                self.discriminator.zero_grad()
                # Obtain aged image from generator
                aged_image = self.generator(torch.cat([source_img_128, true_label_128], dim=1))
                # Calculate loss on all real batch
                d1_logit = self.discriminator(true_label_img, true_label_64)
                d2_logit = self.discriminator(true_label_img, fake_label_64)
                d3_logit = self.discriminator(aged_image.detach(), true_label_64)
                # Do label smoothing and create targets:
                valid_target = torch.ones(d1_logit.shape)
                fake_target = torch.zeros(d1_logit.shape)
                # Calculate all real loss
                d1_real_loss = self.criterion_mse(d1_logit, valid_target.to(self.device))
                # Calculate real image, fake condition loss
                d2_fake_loss = self.criterion_mse(d2_logit, fake_target.to(self.device))
                # Calculate fake image, real condition loss
                d3_fake_loss = self.criterion_mse(d3_logit, fake_target.to(self.device))
                # Calculate the average loss
                d_loss = 0.5 * (d1_real_loss + 0.5 * (d2_fake_loss + d3_fake_loss)) * self.w_gan_loss
                # Calculate gradients wrt all losses
                d_loss.backward()
                # Apply the gradient update
                self.d_optim.step()

                # Train the generator
                self.generator.zero_grad()
                # Get age prediction
                gen_age, gen_features = self.classifier(aged_image)
                _, src_features = self.classifier(source_img_128)
                # Get adversarial loss
                d3_logit = self.discriminator(aged_image, true_label_64)
                # Label Smoothing
                valid_target = torch.ones(d1_logit.shape)
                # Get losses
                d3_real_loss = self.criterion_mse(d3_logit, valid_target.to(self.device)) * self.w_gan_loss
                age_loss = self.criterion_ce(gen_age, true_label) * self.w_age_loss
                feature_loss = self.criterion_mse(gen_features, src_features) * self.w_feat_loss
                # Get avg loss
                g_loss = d3_real_loss + age_loss + feature_loss
                # Backprop the g_loss
                g_loss.backward()
                # Apply update to the generator
                self.g_optim.step()
                # log sampled images
                if step % 200 == 0:
                    print('g_loss', g_loss.item(), 'd_loss', d_loss.item())
                    self.writer.add_scalar('g_loss', g_loss.item(), len(train_queue) * epoch + step)
                    self.writer.add_scalar('d_loss', d_loss.item(), len(train_queue) * epoch + step)
                    grid = torchvision.utils.make_grid(source_img_128,
                                                       normalize=True,
                                                       range=(0, 1),
                                                       scale_each=True)
                    self.writer.add_image('source_image', grid, len(train_queue) * epoch + step)
                    grid = torchvision.utils.make_grid(aged_image,
                                                       normalize=True,
                                                       range=(0, 1),
                                                       scale_each=True)
                    self.writer.add_image('generated_images', grid, len(train_queue) * epoch + step)
                    grid = torchvision.utils.make_grid(true_label_img,
                                                       normalize=True,
                                                       range=(0, 1),
                                                       scale_each=True)
                    self.writer.add_image('target_images', grid, len(train_queue) * epoch + step)
                    self.writer.flush()
                    # Save the weights
                    torch.save(self.generator.state_dict(), 'models/gen.pth')
                    torch.save(self.discriminator.state_dict(), 'models/disc.pth')
