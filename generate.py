import argparse
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from os import path
from model import Generator


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


def save(output_dir, images):
    """
    This function will save all the new images to an output directory.
    :param output_dir: The directory to save the images to.
    :param images: A tensor of generated images.
    :return:
    """
    print("Saving images")
    images = F.interpolate(images, size=SIZE, mode='bicubic')  # The resize operation on tensor.
    for idx, image in enumerate(images):
        name = str(idx) + '.png'
        file_path = path.join(output_dir, name)
        print(image.shape)
        vutils.save_image(image, file_path)
    print("Finished saving images")


def generate(args):
    """
    This is the top level function that will execute the necessary steps to generate the required amount of new images.
    :param args: The arguments passed to the program from the user.
    :return: /
    """
    model_path = args.modelPath
    num = args.num
    output_dir = args.outputDir
    ngpu = args.ngpu
    nz = args.nz
    ngf = args.ngf
    nc = args.nc

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    generator = load_generator(model_path, ngpu, nz, ngf, nc, device)
    images = generate_images(generator, num, nz, device)
    save(output_dir, images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', required=True, help='path to the model checkpoint')
    parser.add_argument('--num', type=int, required=True, help='number of artworks to generate')
    parser.add_argument('--outputDir', required=True, help='The directory to save the generated works to')
    parser.add_argument('--nc', type=int, default=3, help='The number of color channels in the input images.')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use. Use 0 for CPU mode.')
    args = parser.parse_args()

    generate(args)
