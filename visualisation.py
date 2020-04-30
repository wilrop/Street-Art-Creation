import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path


def plot_images(data, epoch=None, output_dir=None):
    """
    This function takes data and plots a grid image from it.
    :param data: The images in the form of tensors.
    :param epoch: What epoch the samples are from. If epoch is None, we assume that they are real training samples.
    :param output_dir: The directory to save the created grid to.
    :return: /
    """
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    if epoch is None:
        plt.title("Sample Training Images")
    else:
        plt.title("Generated Images During {}th Epoch".format(epoch))
    plt.imshow(
        np.transpose(vutils.make_grid(data[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

    if output_dir is not None:
        if epoch is not None:
            filename = "epoch_{}_results.png".format(epoch)
            output_path = path.join(output_dir, filename)
        else:
            output_path = path.join(output_dir, "training_samples.png")
        plt.savefig(output_path)

    plt.show()


def plot_losses(G_losses, D_losses, output_dir=None):
    """
    This function will plot the losses for the generator and discriminator over time.
    :param G_losses: The loss for the generator over the iterations.
    :param D_losses: The loss for the discriminator over the interations.
    :param output_dir: The directory to save the loss plot to.
    :return: /
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    if output_dir is not None:
        output_path = path.join(output_dir, "training_loss.png")
        plt.savefig(output_path)

    plt.show()


def plot_real_fake(real_imgs, gen_imgs, output_dir=None):
    """
    This function will plot a grid comparison of the real images and generated images.
    :param real_imgs: A random sample or training images.
    :param gen_imgs: A batch of generated images by the network.
    :param output_dir: The directory to save the image to.
    :return: /
    """
    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)  # Arguments are nrows, ncolumns, index
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_imgs[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(gen_imgs, (1, 2, 0)))

    if output_dir is not None:
        output_path = path.join(output_dir, "training_results.png")
        plt.savefig(output_path)

    plt.show()

#TODO: Bugfix FFMPEG is missing
def create_animation(img_list):
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]

    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('timelapse.mp4', writer=writer)
