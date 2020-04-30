import argparse
import os
import random
import numpy as np
from PIL import Image
from glob import glob
from visualisation import show_samples

# Hyperparameters
IMAGE_SIZE = 128
NOISE_SIZE = 100
LR_D = 0.00004
LR_G = 0.0004
BATCH_SIZE = 64
EPOCHS = 300
BETA1 = 0.5
WEIGHT_INIT_STDDEV = 0.02
EPSILON = 0.00005
SAMPLES_TO_SHOW = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="where to find the input files", default="/dataset")
    parser.add_argument("--output_dir", help="where to put output files", default="/output")
    parser.add_argument("--image_width", help="the image width after resizing", type=int,  default=128)
    parser.add_argument("--output_height", help="the image height after resizing", type=int,  default=128)
    args = parser.parse_args()

    # Paths
    INPUT_DATA_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Images
    IMAGE_WIDTH = args.image_width
    IMAGE_HEIGHT = args.image_height
    images = []
    for file in glob(INPUT_DATA_DIR + '*'):
        image = Image.open(file).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image_np = np.asarray(image)
        images.append(image_np)

    input_images = np.asarray(images)
    np.random.shuffle(input_images)

    sample_images = random.sample(list(input_images), SAMPLES_TO_SHOW)
    show_samples(sample_images, 0, OUTPUT_DIR, "inputs", IMAGE_WIDTH, IMAGE_HEIGHT)
