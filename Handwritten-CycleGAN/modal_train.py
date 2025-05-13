#!/usr/bin/python3

import modal
import os
import sys
from pathlib import Path
import datetime

# Define the Modal Stub and GPU requirements
app = modal.App("cyclegan-training")
image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "pillow", "tqdm", "matplotlib", "tensorboard"
)

# Define volumes
viz_volume = modal.Volume.from_name("cyclegan-viz", create_if_missing=True)
models_volume = modal.Volume.from_name("cyclegan-models", create_if_missing=True)

# GPU configuration
gpu = modal.gpu.A100()


@app.function(
    image=image,
    gpu=gpu,
    timeout=3600 * 8,  # 8-hour timeout
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/cyclegan")],
    volumes={
        "/root/cyclegan/viz_data": viz_volume,
        "/root/cyclegan/model_output": models_volume,
    },
)
def train_cyclegan(
    epoch=0,
    n_epochs=200,
    batch_size=6,
    dataroot="datasets/horse2zebra/",
    lr=0.0002,
    decay_epoch=100,
    size=128,
    input_nc=3,
    output_nc=3,
    n_cpu=8,
):
    """Run CycleGAN training on Modal"""
    import sys
    import time

    os.chdir("/root/cyclegan")
    sys.path.append("/root/cyclegan")

    # Create a timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{timestamp}"

    # Create output directories with timestamp
    os.makedirs(f"output/{run_dir}", exist_ok=True)
    os.makedirs(f"viz_data/{run_dir}", exist_ok=True)
    os.makedirs(
        f"model_output/{run_dir}", exist_ok=True
    )  # Directory for saving models in volume

    # Import the rest of the modules here to ensure they're in Modal's context
    import torch
    import argparse
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter

    # Create tensorboard writer with run-specific directory
    writer = SummaryWriter(f"viz_data/{run_dir}/tensorboard")

    # Import project files (these will be uploaded via the mount)
    from models import Generator, Discriminator
    from modal_utils import ReplayBuffer, LambdaLR, weights_init_normal

    # Modified Logger to save images periodically
    class FileLogger:
        def __init__(self, n_epochs, batches_epoch, run_dir):
            self.n_epochs = n_epochs
            self.batches_epoch = batches_epoch
            self.epoch = 0
            self.batch = 0
            self.prev_time = time.time()
            self.mean_period = 0
            self.losses = {}
            self.loss_windows = {}
            self.image_windows = {}
            self.run_dir = run_dir

            # Create directories for sample images
            os.makedirs(f"viz_data/{run_dir}/samples", exist_ok=True)

        def log(self, losses=None, images=None):
            self.batch += 1

            # Log losses to tensorboard
            if losses:
                for loss_name, loss in losses.items():
                    if loss_name not in self.losses:
                        self.losses[loss_name] = losses[loss_name]
                    else:
                        self.losses[loss_name] += losses[loss_name]

                    # Log to tensorboard
                    step = self.epoch * self.batches_epoch + self.batch
                    writer.add_scalar(f"loss/{loss_name}", losses[loss_name], step)

            # Save sample images periodically
            if images and self.batch % 100 == 0:
                # Save sample images
                for image_name, tensor in images.items():
                    # Normalize from [-1, 1] to [0, 1] for saving
                    image_numpy = (
                        (tensor[0].data.cpu().numpy().transpose(1, 2, 0) + 1) / 2
                    ) * 255.0
                    image_numpy = image_numpy.clip(0, 255).astype("uint8")

                    # Save the image
                    img_path = f"viz_data/{self.run_dir}/samples/epoch{self.epoch}_batch{self.batch}_{image_name}.png"
                    plt.imsave(img_path, image_numpy)

            # Print progress
            batches_done = self.batches_epoch * self.epoch + self.batch
            batches_left = self.batches_epoch * self.n_epochs - batches_done
            time_left = batches_left * self.mean_period / 3600
            sys.stdout.write(
                f"\r[Epoch {self.epoch}/{self.n_epochs}] [Batch {self.batch}/{self.batches_epoch}] ETA: {time_left:.2f}h"
            )
            sys.stdout.flush()

            # If at the end of an epoch, print epoch summary
            if self.batch == self.batches_epoch:
                # Print epoch losses
                sys.stdout.write(f"\r[Epoch {self.epoch}/{self.n_epochs}] ")
                for loss_name, loss in self.losses.items():
                    sys.stdout.write(f"{loss_name}: {loss/self.batch:.4f} ")
                    self.losses[loss_name] = 0.0
                sys.stdout.write("\n")

                self.epoch += 1
                self.batch = 0
                sys.stdout.flush()

        def reset(self):
            self.losses = {}

    from datasets import ImageDataset

    # Create a parser with the same defaults as the original script
    # but we'll override with the function arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.epoch = epoch
    args.n_epochs = n_epochs
    args.batchSize = batch_size
    args.dataroot = dataroot
    args.lr = lr
    args.decay_epoch = decay_epoch
    args.size = size
    args.input_nc = input_nc
    args.output_nc = output_nc
    args.n_cpu = n_cpu
    args.cuda = True  # Always use GPU on Modal

    print(args)

    # Networks
    netG_A2B = Generator(args.input_nc, args.output_nc)
    netG_B2A = Generator(args.output_nc, args.input_nc)
    netD_A = Discriminator(args.input_nc)
    netD_B = Discriminator(args.output_nc)

    # Move models to GPU
    netG_A2B.to(torch.device("cuda"))
    netG_B2A.to(torch.device("cuda"))
    netD_A.to(torch.device("cuda"))
    netD_B.to(torch.device("cuda"))

    # We'll use a single GPU on Modal, so no need for DataParallel

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    import itertools

    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step,
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A,
        lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step,
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B,
        lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step,
    )

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(args.batchSize, args.input_nc, args.size, args.size)
    input_B = Tensor(args.batchSize, args.output_nc, args.size, args.size)
    target_real = Tensor(args.batchSize).fill_(1.0)
    target_fake = Tensor(args.batchSize).fill_(0.0)

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [
        transforms.Resize(int(args.size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    dataloader = DataLoader(
        ImageDataset(args.dataroot, transforms_=transforms_, unaligned=True),
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.n_cpu,
    )

    # Loss plot
    logger = FileLogger(args.n_epochs, len(dataloader), run_dir)

    # Training loop
    for epoch in range(args.epoch, args.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = input_A.copy_(batch["A"])
            real_B = input_B.copy_(batch["B"])

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            loss_G.backward()
            optimizer_G.step()

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            # Progress report
            logger.log(
                {
                    "loss_G": loss_G,
                    "loss_G_GAN": (loss_GAN_A2B + loss_GAN_B2A),
                    "loss_G_cycle": (loss_cycle_ABA + loss_cycle_BAB),
                    "loss_D": (loss_D_A + loss_D_B),
                },
                images={
                    "real_A": real_A,
                    "real_B": real_B,
                    "fake_A": fake_A,
                    "fake_B": fake_B,
                },
            )

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        if epoch % 20 == 19:
            # Save to local output directory (temporary)
            torch.save(netG_A2B.state_dict(), f"output/{run_dir}/{epoch}_netG_A2B.pth")
            torch.save(netG_B2A.state_dict(), f"output/{run_dir}/{epoch}_netG_B2A.pth")
            torch.save(netD_A.state_dict(), f"output/{run_dir}/{epoch}_netD_A.pth")
            torch.save(netD_B.state_dict(), f"output/{run_dir}/{epoch}_netD_B.pth")

            # Also save to the persistent Modal volume
            torch.save(
                netG_A2B.state_dict(), f"model_output/{run_dir}/{epoch}_netG_A2B.pth"
            )
            torch.save(
                netG_B2A.state_dict(), f"model_output/{run_dir}/{epoch}_netG_B2A.pth"
            )
            torch.save(
                netD_A.state_dict(), f"model_output/{run_dir}/{epoch}_netD_A.pth"
            )
            torch.save(
                netD_B.state_dict(), f"model_output/{run_dir}/{epoch}_netD_B.pth"
            )

    # Save final models to Modal volume
    torch.save(netG_A2B.state_dict(), f"model_output/{run_dir}/final_netG_A2B.pth")
    torch.save(netG_B2A.state_dict(), f"model_output/{run_dir}/final_netG_B2A.pth")
    torch.save(netD_A.state_dict(), f"model_output/{run_dir}/final_netD_A.pth")
    torch.save(netD_B.state_dict(), f"model_output/{run_dir}/final_netD_B.pth")

    # Close tensorboard writer
    writer.close()

    print(f"Training complete! Models saved in run_{timestamp} directory")
    return f"Training completed successfully. Run ID: {timestamp}"


@app.local_entrypoint()
def main():
    # You can customize these parameters when running the script
    params = {
        "epoch": 0,
        "n_epochs": 400,
        "batch_size": 1,
        "dataroot": "datasets/jiannobg_renamed/",
        "lr": 0.0002,
        "decay_epoch": 100,
        "size": 128,
        "input_nc": 3,
        "output_nc": 3,
        "n_cpu": 8,
    }

    # Call the remote function
    result = train_cyclegan.remote(**params)
    print(result)
