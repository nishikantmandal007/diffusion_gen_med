import warnings
from tensorflow.keras.applications import VGG16
from keras.losses import BinaryCrossentropy
from keras.layers import Activation, LeakyReLU, BatchNormalization, Dropout, Resizing
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
import os
import seaborn as sns
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.optimizers import Adam
except:
    from keras.optimizers import Adam

NOISE_DIM = 100
BATCH_SIZE = 4
STEPS_PER_EPOCH = 3759
EPOCHS = 50
SEED = 40
WIDTH, HEIGHT, CHANNELS = 128, 128, 1

OPTIMIZER1 = Adam(0.0002, 0.5)

OPTIMIZER2 = Adam(0.0002, 0.5)

MAIN_DIR = "./input/yes"


def load_images(folder):

    imgs = []
    target = 1
    labels = []
    for i in os.listdir(folder):
        img_dir = os.path.join(folder, i)
        try:
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 128))
            imgs.append(img)
            labels.append(target)
        except:
            continue

    imgs = np.array(imgs)
    labels = np.array(labels)

    return imgs, labels


data, labels = load_images(MAIN_DIR)

np.random.seed(SEED)
idxs = np.random.randint(0, 155, 20)
X_train = data[idxs]

# Normalize the Images
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Reshape images
X_train = X_train.reshape(-1, WIDTH, HEIGHT, CHANNELS)


def build_generator():
    """
        Generator model "generates" images using random noise. The random noise AKA Latent Vector
        is sampled from a Normal Distribution which is given as the input to the Generator. Using
        Transposed Convolution, the latent vector is transformed to produce an image
        We use 3 Conv2DTranspose layers (which help in producing an image using features; opposite
        of Convolutional Learning)

        Input: Random Noise / Latent Vector
        Output: Image
    """

    model = Sequential([

        Dense(32*32*256, input_dim=NOISE_DIM),
        LeakyReLU(alpha=0.2),
        Reshape((32, 32, 256)),

        Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(CHANNELS, (4, 4), padding='same', activation='tanh')
    ],
        name="generator")
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER1)

    return model


def build_discriminator():
    """
        Discriminator is the model which is responsible for classifying the generated images
        as fake or real. Our end goal is to create a Generator so powerful that the Discriminator
        is unable to classify real and fake images
        A simple Convolutional Neural Network with 2 Conv2D layers connected to a Dense output layer
        Output layer activation is Sigmoid since this is a Binary Classifier

        Input: Generated / Real Image
        Output: Validity of Image (Fake or Real)

    """

    model = Sequential([

        Conv2D(64, (3, 3), padding='same',
               input_shape=(WIDTH, HEIGHT, CHANNELS)),
        LeakyReLU(alpha=0.2),

        Conv2D(128, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(128, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(256, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Flatten(),
        Dropout(0.4),
        Dense(1, activation="sigmoid", input_shape=(WIDTH, HEIGHT, CHANNELS))
    ], name="discriminator")
    model.summary()
    model.compile(loss="binary_crossentropy",
                  optimizer=OPTIMIZER2)

    return model


print('\n')
discriminator = build_discriminator()
print('\n')
generator = build_generator()

discriminator.trainable = False

gan_input = Input(shape=(NOISE_DIM,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output, name="gan_model")
opt3 = Adam(0.0002, 0.5)
trainable_vars = gan.trainable_variables
opt3.build(trainable_vars)
gan.compile(loss="binary_crossentropy", optimizer=opt3)

print("The Combined Network:\n")
gan.summary()


if not os.path.exists("./gan_generated_images/"):
    os.makedirs("./gan_generated_images/")
if not os.path.exists("./gan_generated_images/yes/"):
    os.makedirs("./gan_generated_images/yes/")
if not os.path.exists("./gan_generated_images/no/"):
    os.makedirs("./gan_generated_images/no/")


def sample_images(noise, subplots, figsize=(22, 8), save=False):
    generated_images = generator.predict(noise)
    save_folder = "./gan_generated_images/yes/"
    for i, image in enumerate(generated_images):
        # image is
        # save image
        plt.imshow(image.reshape((WIDTH, HEIGHT)), cmap='gray')
        #
        plt.axis('off')
        # plt.show()
        # image is in [-1, 1] , get back to 0-255
        image = (image + 1) * 127.5
        print(image)
        # add color channel
        img = np.array(image)
        print(img.shape)
        img_rgb = np.repeat(img, 3, axis=-1)
        print(img_rgb.shape)
        img_rgb = img_rgb.astype(np.uint8)
        # show image
        plt.imshow(img_rgb)

        Image.fromarray(img_rgb).save(save_folder+"gen"+str(i)+".png")


np.random.seed(SEED)
for epoch in range(EPOCHS):
    for batch in tqdm(range(STEPS_PER_EPOCH)):

        noise = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_DIM))
        fake_X = generator.predict(noise)

        idx = np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)
        real_X = X_train[idx]

        X = np.concatenate((real_X, fake_X))

        disc_y = np.zeros(2*BATCH_SIZE)
        disc_y[:BATCH_SIZE] = 1

        d_loss = discriminator.train_on_batch(X, disc_y)

        y_gen = np.ones(BATCH_SIZE)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(
        f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")
    noise = np.random.normal(0, 1, size=(10, NOISE_DIM))
    # sample_images(noise, (2, 5))


num_images = 2
noise = np.random.normal(0, 1, size=(num_images, NOISE_DIM))
sample_images(noise, (1, num_images), save=True)
