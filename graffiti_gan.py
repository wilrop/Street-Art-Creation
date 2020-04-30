from __future__ import print_function
import argparse
import random
import time
from os import path
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from model import Discriminator, Generator
from util import create_dataloader, weights_init
from visualisation import plot_images, plot_losses, plot_real_fake, create_animation


def load_gan(checkpoint, netG, netD, optimG, optimD):
    checkpoint = torch.load(checkpoint)
    netG.load_state_dict(checkpoint['G_model_state_dict'])
    netD.load_state_dict(checkpoint['D_model_state_dict'])
    optimG.load_state_dict(checkpoint['G_optimizer_state_dict'])
    optimD.load_state_dict(checkpoint['D_optimizer_state_dict'])
    G_losses = checkpoint['G_losses']
    D_losses = checkpoint['D_losses']

    netG.train()
    netD.train()

    return netG, netD, optimG, optimD, G_losses, D_losses


def train(args):
    # Set random seed for reproducibility
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # Creating the dataloader.
    dataloader = create_dataloader(args.dataroot, args.imageSize, args.batchSize, args.workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    real_images, labels = real_batch  # The images in the batch and the labels which are given based on the origin dir.
    plot_images(real_images)

    # Create the generator and discriminator.
    netG = Generator(args.ngpu, args.nz, args.ngf, args.nc).to(device)
    netD = Discriminator(args.ngpu, args.nc, args.ndf).to(device)

    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
        netD = nn.DataParallel(netD, list(range(args.ngpu)))

    if args.checkpoint is not None:
        netG, netD, optimizerG, optimizerD, G_losses, D_losses = load_gan(args.checkpoint, netG, netD, optimizerG,
                                                                          optimizerD)
    else:
        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        netG.apply(weights_init)
        netD.apply(weights_init)

        G_losses = []
        D_losses = []

    # Print the models
    print(netG)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    iters = 0

    print("Starting Training Loop...")
    start = time.time()
    # For each epoch
    for epoch in range(args.epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # Save checkpoints
            filename = "epoch_{}.pth".format(epoch)
            checkpoint_path = path.join(args.outputDir, filename)
            torch.save({
                'epoch': epoch,
                'G_model_state_dict': netG.state_dict(),
                'D_model_state_dict': netD.state_dict(),
                'G_optimizer_state_dict': optimizerG.state_dict(),
                'D_optimizer_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses
            }, checkpoint_path)

            # Plot updated samples.
            fake = netG(fixed_noise).detach().cpu()
            plot_images(fake, epoch)

    end = time.time()
    elapsed = end - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    plot_losses(G_losses, D_losses, args.outputDir)
    plot_real_fake(real_images, img_list[-1], args.outputDir)
    # create_animation(img_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='The number of color channels in the input images.')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use. Use 0 for CPU mode.')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outputDir', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--checkpoint', help='File to load checkpoints from')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()

    train(args)
