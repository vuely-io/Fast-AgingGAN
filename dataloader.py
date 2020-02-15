import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import read_image_label_txt, read_image_label_pair_txt


class DataLoaderAge(Dataset):
    """DataSet for the age classifier."""

    def __init__(self, image_dir, text_dir, image_size, is_train=True):
        """Initializes the data loader for training the age classifier
        Args:
            image_dir: str, path to directory where images are located.
            text_dir: str, path to the directory with data split txt files.
            image_size: tuple, (Height, Width), size of images to train on.
        """
        self.image_labels, self.image_paths = read_image_label_pair_txt(image_dir, text_dir, is_train)
        self.image_size = image_size
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx: The index of the image and label to read.
        """
        choice = random.randint(0, 1)
        image = Image.open(self.image_paths[idx][choice])
        image = image.resize(self.image_size)
        age = int(self.image_labels[idx][choice])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, age


class DataLoaderGAN(Dataset):
    """Dataset for the face aging GAN"""

    def __init__(self, image_dir, text_dir, image_size, is_train=True):
        """
        Args:
            image_dir: str, path to the directory with face images.
            text_dir: str, path to the directory with data split text files.
            image_size: tuple (H, W), the spatial size of the image.
                image.
        """
        self.source_images, _ = read_image_label_txt(image_dir, text_dir)
        self.label_pairs, self.image_pairs = read_image_label_pair_txt(image_dir, text_dir, is_train)
        self.image_size = image_size
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def _condition_images(self, source_image, true_label, false_label):
        """
        Args:
            source_image: tensor, torch tensor of the input image to the
                generator.
            true_label: int, the integer category label of the target domain.
            false_label: int, the integer category label of the non-target
                domain images.
        """
        h, w = self.image_size
        true_condition = torch.ones((1, h, w)) * true_label
        false_condition = torch.ones([1] + [x // 2 for x in self.image_size]) * false_label

        source_image_conditioned = torch.cat([source_image, true_condition], dim=0)
        true_condition = torch.ones([1] + [x // 2 for x in self.image_size]) * true_label

        return source_image_conditioned, true_condition, false_condition

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of items to retrieve.
        """
        source_image = Image.open(self.source_images[idx]).resize(self.image_size)
        true_image = Image.open(self.image_pairs[idx][0]).resize(self.image_size)

        true_label = self.label_pairs[idx][0]
        false_label = self.label_pairs[idx][1]

        if self.transforms is not None:
            source_image = self.transforms(source_image)
            true_image = self.transforms(true_image)

        src_image_cond, true_condition, false_condition = self._condition_images(source_image, true_label, false_label)

        dataset = {'src_image_cond': src_image_cond,
                   'true_image': true_image,
                   'true_cond': true_condition,
                   'false_cond': false_condition,
                   'true_label': true_label}
        return dataset
