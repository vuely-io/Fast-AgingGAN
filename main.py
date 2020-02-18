from argparse import ArgumentParser

from pytorch_lightning import Trainer

from adversarial_setup import AgeModule, GenAdvNet

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, default='/Users/hasnainraza/Datasets/CACD2000',
                    help='Path to face image directory.')
parser.add_argument('--text_dir', default='data_split',
                    type=str, help='Path to face image directory.')
parser.add_argument('--train_classifier', action='store_true',
                    help='Whether to train the age classifier or use a checkpoint')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs for training')
parser.add_argument('--image_size', default=128, type=int, help='Face image input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')


def main():
    args = parser.parse_args()

    trainer = Trainer(max_epochs=20, gpus=[1])
    if args.train_classifier:
        age_classifier = AgeModule(image_dir=args.image_dir,
                                   text_dir=args.text_dir,
                                   image_size=args.image_size,
                                   batch_size=args.batch_size * 4)
        trainer.fit(age_classifier)

    trainer = Trainer(max_epochs=args.epochs, gpus=[1], early_stop_callback=False)
    gan = GenAdvNet(image_dir=args.image_dir,
                    text_dir=args.text_dir,
                    image_size=args.image_size,
                    batch_size=args.batch_size)
    trainer.fit(gan)


if __name__ == '__main__':
    main()
