import argparse
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from os import path
from model import Generator
from PIL import Image
from skimage.metrics import structural_similarity as ssim


SIZE = 256


def load_generator(path, ngpu, nz, ngf, nc, device):
    """
    This function will load the generator with the given parameters from a checkpoint.
    :param path: The path to the checkpoint.
    :param ngpu: The amount of GPUs to use.
    :param nz: The size of the latent vector.
    :param ngf: The size of the feature maps in the generator.
    :param nc: The number of color channels.
    :param device: The device to run the computations on.
    :return: The instantiated generator.
    """
    print("Loading generator")
    generator = Generator(ngpu, nz, ngf, nc).to(device)
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['G_model_state_dict'])
    generator.eval()
    print("Finished loading generator")
    return generator


def generate_images(generator, num, nz, device):
    """
    This function will use the generator to generate new images.
    :param generator: The generator model.
    :param num: The number of images to generate.
    :param nz: The latent vector size.
    :param device: The device to run the computations on.
    :return: The generated images.
    """
    print("Generating images")
    noise = torch.randn(num, nz, 1, 1, device=device)
    with torch.no_grad():
        images = generator(noise).cpu()
        return images


def get_list_files(dir):
    """
    This procedure will recursively get all the files starting from a top level directory.
    :param dir: The top level directory.
    :return: All files in this directory and any subdirectory in it.
    """
    list_files = os.listdir(dir)
    files = []
    for entry in list_files:
        full_path = os.path.join(dir, entry)
        if os.path.isdir(full_path):
            files = files + get_list_files(full_path)
        else:
            files.append(full_path)
    return files


def plot_imgs(img, parent, sim):
    """
    This procedure will plot the two most similar images next to eachother.
    :param img: The generated image.
    :param parent: The parent image.
    :param sim: The structural similarity between the two.
    :return: /
    """
    sim = sim * 100
    sim = "{:.2f}".format(sim)
    fig = plt.figure()
    fig.suptitle("Similarity: " + sim + "%")

    plt.subplot(1, 2, 1)  # Arguments are nrows, ncolumns, index
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(np.asarray(img))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Parent")
    plt.imshow(np.asarray(parent))
    plt.show()


def save(output_dir, images, csv, parents_dir):
    """
    This function will save all the new images to an output directory.
    :param output_dir: The directory to save the images to.
    :param images: A tensor of generated images.
    :param csv: The CSV to save the results to.
    :param parents_dir: The directory that contains all the parents for the generated images.
    :return: /
    """
    print("Saving images")
    images = F.interpolate(images, size=SIZE, mode='bicubic')  # The resize operation on tensor.

    csv_exists = path.exists(csv)
    if csv_exists:
        data = pd.read_csv(csv)
        idx_offset = len(data)
    else:
        idx_offset = 0

    parents = get_list_files(parents_dir)

    WIDTH = 64
    HEIGHT = 64

    # We will loop over all images and save them. Next we find its closest parent and write it to a dataframe.
    data = {"Path": [], "Parent Path": [], "Similarity": []}
    for idx, image in enumerate(images):
        name = str(idx_offset + idx) + '.png'
        img_path = path.join(output_dir, name)
        vutils.save_image(image, img_path)
        data["Path"].append(img_path)

        print("Searching parent for image " + str(idx + 1) + " out of " + str(len(images)))
        img = Image.open(img_path)
        img_res = img.resize((WIDTH, HEIGHT))
        img_array = np.asarray(img_res)
        best = (None, None, -2)

        # Inner loop to go over all possible parents.
        for parent_path in parents:
            parent = Image.open(parent_path)
            if parent.mode != 'RGB':
                parent = parent.convert('RGB')
            parent_res = parent.resize((WIDTH, HEIGHT))
            parent_array = np.asarray(parent_res)
            score = ssim(img_array, parent_array, multichannel=True)
            if score > best[2]:
                best = (parent_path, parent, score)
        print("Found parent")
        data["Parent Path"].append(best[0])
        data["Similarity"].append(best[2])
        plot_imgs(img, best[1], best[2])

    df = pd.DataFrame(data)
    if csv_exists:
        df.to_csv(csv, mode='a', header=False, index=False)
    else:
        file_path = path.join(output_dir, csv)
        df.to_csv(file_path, index=False)
    print("Finished saving images")


def generate(args):
    """
    This is the top level function that will execute the necessary steps to generate the required amount of new images.
    :param args: The arguments passed to the program from the user.
    :return: /
    """
    model_path = args.modelPath
    parents_dir = args.parentsDir
    num = args.num
    output_dir = args.outputDir
    artCSV = args.artCSV
    ngpu = args.ngpu
    nz = args.nz
    ngf = args.ngf
    nc = args.nc

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    generator = load_generator(model_path, ngpu, nz, ngf, nc, device)
    images = generate_images(generator, num, nz, device)
    save(output_dir, images, artCSV, parents_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', required=True, help='path to the model checkpoint')
    parser.add_argument('--parentsDir', required=True, help='Path to the directory that holds all the parent images')
    parser.add_argument('--num', type=int, required=True, help='number of artworks to generate')
    parser.add_argument('--outputDir', required=True, help='The directory to save the generated works to')
    parser.add_argument('--artCSV', default='artworks.csv', help='The current CSV holding all the works and their features')
    parser.add_argument('--nc', type=int, default=3, help='The number of color channels in the input images.')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use. Use 0 for CPU mode.')
    args = parser.parse_args()

    generate(args)
