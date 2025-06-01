# scripts/crop_images.py
from PIL import Image
import os

def crop_image(input_path, output_path, crop_size):
    image = Image.open(input_path)
    width, height = image.size
    left = (width - crop_size) // 2
    upper = (height - crop_size) // 2
    right = left + crop_size
    lower = upper + crop_size
    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, required=True)
    args = parser.parse_args()
    crop_image(args.input, args.output, args.size)
