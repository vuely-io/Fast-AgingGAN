from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str,
                    help='Path to face image directory.')
parser.add_argument('--text_dir', default='data_split', type=str,
                    help='Path to face image directory.')
parser.add_argument('--batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs for training')
parser.add_argument('--img_size', default=128, type=int, help='Face image input size.')
parser.add_argument('--num_classes', default=5, type=int, help='Number of age categories')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')


def main():
    args = parser.parse_args()
    pass


if __name__ == '__main__':
    main()
