from tensorflow import keras
import tensorflow as tf


class AgingGAN(object):
    """Aging GAN for faces."""

    def __init__(self, args, age_train=False):
        """
        Initializes the Fast AgingGAN class.
        Args:
            args: CLI arguments that dictate how to build the model.
            age_train: Whether to train the age classifer or use an existing one.
        Returns:
            None
        """
        self.img_dim = args.img_size
        self.img_size = (args.img_size, args.img_size, 3)
        self.iterations = 0

        # Number of inverted residual blocks in the generator
        self.n_residual_blocks = 6

        # Define Optimizers
        self.gen_optimizer = keras.optimizers.Adam(args.lr)
        self.disc_optimizer = keras.optimizers.Adam(args.lr)
        self.cls_optimizer = keras.optimizers.Adam(args.lr)

        # Calculate output shape of D (PatchGAN)
        patch = int(args.img_size / 2 ** 5)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32  # Realtime Image Enhancement GAN Galteri et al.
        self.df = 64

        # If training age classifier, load only that into memory
        if age_train:
            self.age_classifier = self.build_age_classifier(args.num_classes)
        else:
            # Otherwise load the GAN setup
            self.age_classifier = keras.models.load_model('models/age_classifier.h5')
            self.age_classifier.trainable = False
            for layer in self.age_classifier.layers:
                layer.trainable = False

            # Build feature extractor
            self.feature_extractor = self.build_feat_extractor()

            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()

            # Build and compile the generator
            self.generator = self.build_generator()

    @tf.function
    def classifier_loss(self, fake, age_labels, weight=30):
        """
        The content loss for the generator for face aging.
        Args:
            fake: The generated target domain image.
            age_labels: The age class labels for the classifier loss.
            weight: Weight given to the loss.
        Returns:
            loss: tf tensor of the sum of feature MSE and age classifier loss.
        """
        fake = (fake + 1.0) / 2.0
        fake_labels = self.age_classifier(fake)
        age_loss = weight * tf.keras.losses.SparseCategoricalCrossentropy()(age_labels, fake_labels)
        return age_loss

    @tf.function
    def feature_loss(self, real, fake, weight=5e-5):
        """
        The feature loss for the face Aging, preserves identity
        and is also called the Identity Preserving Module
        Args:
            real: The image that was input to the model.
            fake: The generated aged image the generator gave.
            weight: Weight given to the loss.

        """
        fake = keras.applications.vgg16.preprocess_input((fake + 1.0) / 2.0)
        real = keras.applications.vgg16.preprocess_input((real + 1.0) / 2.0)
        fake_features = self.feature_extractor(fake)
        real_features = self.feature_extractor(real)
        feature_loss = weight * tf.keras.losses.MeanSquaredError()(real_features, fake_features)
        return feature_loss

    def build_age_classifier(self, num_classes):
        """
        Builds a pre-trained VGG network for image classification
        Args:
            num_classes: The number of classes for the classifier.
        Returns:
            model: A tf keras model for the classifier.
        """
        # Input image to extract features from
        inputs = keras.Input((self.img_dim, self.img_dim, 3))
        features = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=self.img_size)(inputs)
        x = keras.layers.Flatten()(features)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)

        # Compile the model
        model = keras.models.Model(inputs, x)

        return model

    def build_feat_extractor(self):
        """
        Builds a VGG16 feature extractor, extracts features from the last conv layer in vgg16.
        """
        # Get the vgg network. Extract features from Block 5, last convolution.
        vgg = keras.applications.VGG19(weights="imagenet", input_shape=(self.img_dim, self.img_dim, 3),
                                       include_top=False)
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False

        # Create model and compile
        model = keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

        return model

    def build_generator(self):
        """Build the generator that will do the Face Aging task."""

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def residual_block(inputs, filters):
            """Inverted Residual block that uses depth wise convolutions for parameter efficiency.
            Args:
                inputs: The input feature map.
                filters: Number of filters in each convolution in the block.
            Returns:
                x: The output of the inverted residual block.
            """
            u = keras.layers.Conv2D(filters, kernel_size=3, stride=1, padding='same')(inputs)
            u = keras.layers.BatchNormalization()(u)
            u = keras.layers.Conv2D(filters, kernel_size=3, stride=1, padding='same')(u)
            u = keras.layers.BatchNormalization()(u)
            u = keras.layers.Add()([inputs, u])
            return u

        def deconv2d(layer_input, filters):
            """Upsampling layer to increase height and width of the input.
            Uses PixelShuffle for upsampling.
            Args:
                layer_input: The input tensor to upsample.
                filters: Numbers of expansion filters.
            Returns:
                u: Upsampled input by a factor of 2.
            """
            u = keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(layer_input)
            u = keras.layers.LeakyReLU()(u)
            return u

        # Original image input
        img_lr = keras.Input(shape=(self.img_dim, self.img_dim, 4))

        # Pre-residual block
        x = keras.layers.Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(img_lr)
        x = keras.layers.BatchNormalization()(x)
        c1 = keras.layers.LeakyReLU()(x)

        # Downsample
        x = keras.layers.Conv2D(self.gf, kernel_size=3, strides=2, padding='same')(c1)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        # Downsample
        x = keras.layers.Conv2D(self.gf, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        # Propogate through residual blocks
        for idx in range(0, self.n_residual_blocks):
            x = residual_block(x, self.gf, idx)

        # Upsampling
        x = deconv2d(x, self.gf)
        x = deconv2d(x, self.gf)

        # Generate output
        gen_hr = keras.layers.Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return keras.models.Model(img_lr, gen_hr)

    def build_discriminator(self):
        """Builds a discriminator network based on the Patch-GAN design."""

        def d_block(layer_input, filters, strides=1, bn=True, act=True):
            """Discriminator layer block.
            Args:
                layer_input: Input feature map for the convolutional block.
                filters: Number of filters in the convolution.
                strides: The stride of the convolution.
                bn: Whether to use batch norm or not.
            """
            d = keras.layers.Conv2D(filters, kernel_size=4, strides=strides, padding='same')(layer_input)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            if act:
                d = keras.layers.LeakyReLU(alpha=0.2)(d)

            return d

        # Input img
        d0 = keras.layers.Input(shape=self.img_size)
        # Input input condition
        cond = keras.layers.Input(shape=(self.img_dim // 2, self.img_dim // 2, 1))

        d1 = d_block(d0, self.df, strides=2, bn=False)
        d1 = keras.layers.Concatenate()([d1, cond])
        d2 = d_block(d1, self.df * 2, strides=2)
        d3 = d_block(d2, self.df * 4, strides=2)
        d4 = d_block(d3, self.df * 4, strides=2)
        d5 = d_block(d4, 1, strides=2, bn=False, act=False)

        return keras.models.Model([d0, cond], d5)
