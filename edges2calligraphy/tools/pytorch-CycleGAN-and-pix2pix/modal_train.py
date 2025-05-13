"""Modal-specific training script for image-to-image translation models.

This script adapts the original CycleGAN and pix2pix training script to work with Modal for distributed training.
It supports various models (pix2pix, cyclegan, colorization) and different datasets (aligned, unaligned, single, colorization).

Example usage:
    python modal_train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    python modal_train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

With Modal volume:
    python modal_train.py --volume_name cyclegan-facades --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
"""

import os
import time
import modal
from modal import Image, Stub, method, Mount, Secret, Volume

# Define the Modal image with PyTorch and other dependencies
image = Image.debian_slim().pip_install(
    "torch>=1.4.0",
    "torchvision>=0.5.0",
    "dominate>=2.4.0",
    "visdom>=0.1.8.8",
    "wandb",
    "tqdm",
)

# Create a Modal Stub
stub = Stub("pix2pix-cyclegan-trainer")

# Mount the local code directory to Modal
local_dir = os.path.dirname(os.path.abspath(__file__))
mount = Mount.from_local_dir(local_dir, remote_path="/code")


@stub.function(
    image=image,
    gpu="T4",  # Adjust based on your GPU requirements (T4, A10G, A100, etc.)
    mounts=[mount],
    timeout=86400,  # 24 hours max runtime
    secrets=[Secret.from_name("wandb")] if os.environ.get("USE_WANDB") else [],
)
def train_model(args_list, volume_name=None, dataset_dir=None):
    """Train a model with the given arguments passed as a list."""
    import sys

    sys.path.append("/code")

    # Import the required modules from the original codebase
    from options.train_options import TrainOptions
    from data import create_dataset
    from models import create_model
    from util.visualizer import Visualizer

    # Modify arguments for Modal environment if needed
    modified_args = args_list.copy()

    # Add Modal-specific arguments
    modified_args.extend(["--modal_data_mount"])

    # If using persistent volume for data, adjust paths
    if dataset_dir:
        # Update the dataroot argument to point to the mounted volume
        for i, arg in enumerate(modified_args):
            if arg == "--dataroot" and i < len(modified_args) - 1:
                if not modified_args[i + 1].startswith("/"):
                    modified_args[i + 1] = os.path.join(
                        dataset_dir, os.path.basename(modified_args[i + 1])
                    )
                break

    # Create output directory for checkpoints and results
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    modified_args.extend(["--modal_output_dir", output_dir])

    # Parse arguments using the original options parser
    opt = TrainOptions().parse(modified_args)

    # Create dataset, model, and visualizer
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    # Training loop - same as original train.py but with progress tracking
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()

        print(f"Starting epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}")

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # Display images
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data
                )
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )

            # Save latest model
            if total_iters % opt.save_latest_freq == 0:
                print(
                    f"Saving the latest model (epoch {epoch}, total_iters {total_iters})"
                )
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Save model at the end of epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f"Saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            f"End of epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec"
        )

    # Return information about the training run
    return {
        "total_epochs": opt.n_epochs + opt.n_epochs_decay,
        "completed_epochs": epoch,
        "model_name": opt.name,
        "dataset_size": dataset_size,
        "batch_size": opt.batch_size,
        "training_time": time.time() - epoch_start_time,
    }


@stub.local_entrypoint()
def main():
    """Local entrypoint to run the Modal function."""
    import sys
    import argparse

    # Create parser for Modal-specific arguments
    parser = argparse.ArgumentParser(description="Train CycleGAN/pix2pix on Modal")
    parser.add_argument(
        "--volume_name",
        type=str,
        default=None,
        help="Name of Modal volume containing the dataset",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="T4",
        choices=["T4", "A10G", "A100", "H100"],
        help="GPU type to use on Modal",
    )

    # Parse known args for Modal-specific options
    modal_args, remaining_args = parser.parse_known_args()

    # Configure volumes if specified
    volumes = []
    dataset_mount_path = None

    if modal_args.volume_name:
        try:
            volume = Volume.from_name(modal_args.volume_name)
            dataset_mount_path = "/datasets"
            volumes = [(volume, dataset_mount_path)]
            print(f"Using Modal volume: {modal_args.volume_name}")
        except Exception as e:
            print(f"Error accessing volume '{modal_args.volume_name}': {e}")
            print("Continuing without volume mounting")

    # Update GPU type if specified
    train_model.gpu = modal_args.gpu_type

    # If no remaining arguments are provided, show help message
    if not remaining_args:
        print(
            "Usage example: modal run modal_train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA"
        )
        print(
            "With volume: modal run modal_train.py --volume_name cyclegan-facades --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA"
        )
        print("Run with --help for more options")
        sys.exit(1)

    # Run the training function with volumes if specified
    if volumes:
        # Update the function config to include volumes
        train_model.update(volumes=volumes)
        result = train_model.remote(
            remaining_args, modal_args.volume_name, dataset_mount_path
        )
    else:
        result = train_model.remote(remaining_args)

    # Print the result
    print("\n=== Training Complete ===")
    print(f"Model: {result['model_name']}")
    print(f"Completed {result['completed_epochs']}/{result['total_epochs']} epochs")
    print(f"Dataset size: {result['dataset_size']} images")
    print(f"Batch size: {result['batch_size']}")
    print(f"Training time: {result['training_time']:.2f} seconds")
