# dcgan/train.py
import tensorflow as tf
from dcgan.model import build_discriminator, build_generator
from dcgan.dataset import tf_dataset
from utils.plot_utils import save_plot
import numpy as np
import os

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        # Train Discriminator
        for _ in range(2):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))
            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            labels = tf.ones((batch_size, 1))
            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            misleading_labels = tf.ones((batch_size, 1))
            with tf.GradientTape() as gtape:
                predictions = self.discriminator(self.generator(random_latent_vectors))
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = gtape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}

if __name__ == "__main__":
    batch_size = 128
    latent_dim = 128
    num_epochs = 100
    image_data_folder = "data/augmented/images"  
    images_path = [os.path.join(image_data_folder, f) for f in os.listdir(image_data_folder)]
    d_model = build_discriminator()
    g_model = build_generator(latent_dim)
    gan = GAN(d_model, g_model, latent_dim)
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)
    images_dataset = tf_dataset(images_path, batch_size)
    for epoch in range(num_epochs):
        gan.fit(images_dataset, epochs=1)
        g_model.save("samples/g_model.h5")
        d_model.save("samples/d_model.h5")
        if (epoch + 1) % 10 == 0:
            n_samples = 25
            noise = np.random.normal(size=(n_samples, latent_dim))
            examples = g_model.predict(noise)
            save_plot(examples, epoch, int(np.sqrt(n_samples)))
