# dcgan/evaluation.py
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow.keras.applications.inception_v3 import InceptionV3
from scipy.stats import entropy

def calculate_fid(real_images, generated_images, batch_size, model=None):
    if model is None:
        model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    real_features = model.predict(tf.image.resize(real_images, (299, 299)))
    gen_features = model.predict(tf.image.resize(generated_images, (299, 299)))
    m1, m2 = np.mean(real_features, axis=0), np.mean(gen_features, axis=0)
    s1, s2 = np.cov(real_features, rowvar=False), np.cov(gen_features, rowvar=False)
    diff = np.sum((m1 - m2)**2) + np.trace(s1 + s2 - 2*np.dot(np.sqrt(np.dot(s1, s2)), s1))
    return np.sqrt(diff)

def calculate_inception_score(images, inception_model, batch_size=32, num_splits=10, epsilon=1e-10):
    preds = []
    for i in range(num_splits):
        subset = images[i * (len(images) // num_splits):(i + 1) * (len(images) // num_splits)]
        pred = inception_model.predict(subset)
        preds.append(pred)
    preds = np.concatenate(preds)
    p_yx = np.mean(preds, axis=0)
    kl_divs = np.sum(preds * (np.log(preds + epsilon) - np.log(p_yx + epsilon)), axis=1)
    inception_score = np.exp(np.mean(kl_divs))
    return inception_score
