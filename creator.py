import argparse
import random
import pandas as pd
import numpy as np
from os import path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


WIDTH = 64
HEIGHT = 64


def extract_features(image):
    """
    This function will extract features from a given image. As of right now, this function is incomplete and simply
    returns a list of all colors found in the image and their counts.
    :param image: The input image.
    :return: A list of colors and their count.
    """
    print('Extracting features')
    image = image.resize((WIDTH, HEIGHT))
    colors = image.getcolors(WIDTH * HEIGHT)
    features = colors
    print('Finished extracting features')
    return features


def select_artwork(wall, art_csv, features):
    print('Selecting artwork')
    artworks = pd.read_csv(art_csv)

    def select_with_features(artworks, features):
        selection = artworks
        for feature in features:
            selection = selection[selection[feature] != 0]

        if selection.empty:
            to_remove = random.choice(features)
            features = features.remove(to_remove)
            select_with_features(artworks, features)
        else:
            return selection

    # selection = select_with_features(artworks, features)

    def select_with_sim(wall):
        wall_res = wall.resize((WIDTH, HEIGHT))
        wall_array = np.asarray(wall_res)
        best = (None, -2)
        for path in artworks["Path"]:
            image = Image.open(path)
            image_res = image.resize((WIDTH, HEIGHT))
            art_array = np.asarray(image_res)
            score = ssim(wall_array, art_array, multichannel=True)
            if score > best[1]:
                best = (image, score)
        return best[0]

    artwork = select_with_sim(wall)
    print('Finished selecting artwork')
    return artwork


def blend(wall, artwork):
    print('Blending artwork into the wall')
    art_width, art_height = artwork.size
    wall_width, wall_height = wall.size

    art_mid_x = art_width / 2
    art_mid_y = art_height / 2
    wall_mid_x = wall_width / 2
    wall_mid_y = wall_height / 2
    x = int(wall_mid_x - art_mid_x)
    y = int(wall_mid_y - art_mid_y)

    def fade_in(x, y, wall_section, art_section, alpha, final_alpha, final_img, alpha_step=0.02, crop_step=3):
        blend = Image.blend(art_section, wall_section, alpha=alpha)
        final_img.paste(blend, (x, y))
        new_alpha = alpha - alpha_step
        if new_alpha < final_alpha:
            return final_img
        else:
            width, height = art_section.size
            width -= crop_step
            height -= crop_step
            x += crop_step
            y += crop_step
            new_art_section = art_section.crop((crop_step, crop_step, width, height))
            new_wall_section = wall_section.crop((crop_step, crop_step, width, height))
            return fade_in(x, y, new_wall_section, new_art_section, new_alpha, final_alpha, final_img)
    wall_section = wall.crop((x, y, x + art_width, y + art_height))
    final_alpha = 0.4
    faded_img = fade_in(0, 0, wall_section, artwork, 1, final_alpha, wall_section)

    wall.paste(faded_img, (x, y))
    print('Finished blending the artwork into the wall')
    return wall


def show_and_save(street_art, output_dir, output_name):
    print('Showing and saving generated street art')
    plt.axis("off")
    plt.imshow(street_art)
    plt.show()

    fp = path.join(output_dir, output_name)
    street_art.save(fp)
    print('Finished showing and saving street art')


def create(args):
    input_file = args.inputFile
    wall = Image.open(input_file)
    features = extract_features(wall)

    art_csv = args.artCSV
    artwork = select_artwork(wall, art_csv, features)
    street_art = blend(wall, artwork)

    output_dir = args.outputDir
    output_name = args.outputName
    show_and_save(street_art, output_dir, output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', required=True, help='path to input file')
    parser.add_argument('--outputDir', default='.', help='Path to output directory')
    parser.add_argument('--outputName', default='result.png', help='The name for the resulting street art')
    parser.add_argument('--artCSV', required=True, help='path to a CSV containing data for pre-generated works of art')
    args = parser.parse_args()

    create(args)
