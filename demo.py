import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models import MobileGenerator

parser = ArgumentParser()
parser.add_argument('--image_dir', default='/Users/hasnainraza/Datasets/CACD2000', help="Directory of images")
parser.add_argument('--image_size', default=512, help="Size of the images to process")


def main():
    args = parser.parse_args()
    model = MobileGenerator(6)
    model.load_state_dict(torch.load('models/gen.pth', map_location='cpu'))
    model.eval()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]
    random.shuffle(image_paths)

    for step, path in enumerate(image_paths):
        print(f'Inferring on image number {step}')
        fig, ax = plt.subplots(1, 6, figsize=(20, 7))
        image = np.array(Image.open(path).resize((args.image_size, args.image_size))) / 255.0
        h, w = image.shape[:2]
        ax[0].imshow(image)
        with torch.no_grad():
            for label in range(5):
                condition = np.ones((h, w, 1)) * label
                cond_image = np.concatenate([image, condition], axis=2)
                cond_image = torch.from_numpy(cond_image).permute(2, 0, 1).unsqueeze(dim=0)
                aged_image = model(cond_image.float())[0].permute(1, 2, 0)
                ax[label + 1].imshow(aged_image)
            plt.savefig(f'{step}.png')
            plt.close('all')


if __name__ == '__main__':
    main()
