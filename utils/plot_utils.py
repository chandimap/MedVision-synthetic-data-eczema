# utils/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os

def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    if not os.path.exists('samples'):
        os.makedirs('samples')
    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i])
    filename = f"samples/generated_plot_epoch-{epoch+1}.png"
    plt.savefig(filename)
    plt.close()
