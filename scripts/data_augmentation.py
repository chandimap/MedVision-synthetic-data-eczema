# scripts/data_augmentation.py
import Augmentor
import os

def augment_images(input_dir, num_samples):
    p = Augmentor.Pipeline(input_dir)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.crop_random(probability=1, percentage_area=0.5)
    p.sample(num_samples)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--samples', type=int, default=300)
    args = parser.parse_args()
    augment_images(args.input_dir, args.samples)
