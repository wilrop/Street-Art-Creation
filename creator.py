import argparse
import random
import pandas as pd
from PIL import Image


WIDTH = 64
HEIGHT = 64


def extract_features(image):
    image = image.resize((WIDTH, HEIGHT))
    colors = image.getcolors(WIDTH * HEIGHT)
    print(colors)
    features = colors
    return features


def select_artwork(artwork_file, features):
    artworks = pd.read_csv(artwork_file)

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

    selection = select_with_features(artworks, features)

    artwork = random.choice(selection)
    image = Image.open(artwork["Path"])
    return image


def blend(wall, artwork):
    return wall


def create(args):
    input_file = args.inputFile
    wall = Image.open(input_file)
    features = extract_features(wall)

    artwork_file = args.artworkFile
    artwork = select_artwork(artwork_file, features)
    street_art = blend(wall, artwork)
    return street_art


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', required=True, help='path to input file')
    parser.add_argument('--artworkFile', required=True, help='path to pre-generated works of art')
    args = parser.parse_args()

    create(args)
