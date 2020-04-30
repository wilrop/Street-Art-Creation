import argparse
import os
from PIL import Image


def convert_bw(args):
    top_input_folder = args.inputFolder
    top_output_folder = args.outputFolder

    print("Beginning conversion")
    for dir in os.listdir(top_input_folder):
        input_folder = os.path.join(top_input_folder, dir)
        output_folder = os.path.join(top_output_folder, dir)
        os.mkdir(output_folder)
        for idx, file in enumerate(os.listdir(input_folder)):
            path = os.path.join(input_folder, file)
            image = Image.open(path)
            bw_image = image.convert('L')  # L means conversion to grayscale.
            name = str(idx) + '.png'
            output_path = os.path.join(output_folder, name)
            bw_image.save(output_path)

    print("Finished conversion")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFolder', required=True, help='path to input folder that contains all the subdirectories.')
    parser.add_argument('--outputFolder', required=True, help='path to the output folder')
    args = parser.parse_args()

    convert_bw(args)
