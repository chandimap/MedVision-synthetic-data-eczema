# scripts/background_removal.py
from rembg import remove
from PIL import Image

def remove_bg(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input_data = i.read()
            output = remove(input_data)
            o.write(output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    remove_bg(args.input, args.output)
