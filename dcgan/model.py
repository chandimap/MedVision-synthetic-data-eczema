# dcgan/model.py
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

IMG_H, IMG_W, IMG_C = 64, 64, 3
w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = Conv2DTranspose(num_filters, kernel_size, kernel_initializer=w_init, padding="same", strides=strides, use_bias=False)(inputs)
    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    return x

def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(num_filters, kernel_size, kernel_initializer=w_init, padding=padding, strides=strides)(inputs)
    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
    return x

def build_generator(latent_dim):
    f = [2**i for i in range(5)][::-1]
    filters = 32
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides
    noise = Input(shape=(latent_dim,))
    x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((h_output, w_output, 16 * filters))(x)
    for i in range(1, 5):
        x = deconv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)
    x = conv_block(x, num_filters=3, kernel_size=5, strides=1, activation=False)
    fake_output = Activation("tanh")(x)
    return Model(noise, fake_output, name="generator")

def build_discriminator():
    f = [2**i for i in range(4)]
    image_input = Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64
    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i]*filters, kernel_size=5, strides=2)
    x = Flatten()(x)
    x = Dense(1)(x)
    return Model(image_input, x, name="discriminator")
