#!/usr/bin/python3

import os
import argparse
import subprocess
import time
import webbrowser
from pathlib import Path


def download_modal_viz_data(output_dir="./viz_data"):
    """Download visualization data from Modal volume"""
    print(f"Downloading visualization data from Modal volume to {output_dir}...")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Run the Modal command to download data
        subprocess.run(
            ["modal", "volume", "get", "cyclegan-viz", "--dest", output_dir], check=True
        )
        print("✅ Download completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading data: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def start_tensorboard(log_dir):
    """Start TensorBoard server"""
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        print(f"❌ TensorBoard logs directory not found at {tensorboard_dir}")
        return None

    print(f"Starting TensorBoard server for logs in {tensorboard_dir}...")
    try:
        # Start TensorBoard as a subprocess
        process = subprocess.Popen(
            ["tensorboard", "--logdir", tensorboard_dir, "--port", "6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give TensorBoard time to start
        time.sleep(3)

        # Check if the process is still running
        if process.poll() is None:
            print("✅ TensorBoard server started successfully")
            print("📊 View TensorBoard at http://localhost:6006")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ TensorBoard failed to start: {stderr.decode('utf-8')}")
            return None
    except FileNotFoundError:
        print(
            "❌ TensorBoard not found. Please install it with 'pip install tensorboard'"
        )
        return None
    except Exception as e:
        print(f"❌ Error starting TensorBoard: {e}")
        return None


def display_samples_info(viz_dir):
    """Display information about sample images"""
    samples_dir = os.path.join(viz_dir, "samples")
    if not os.path.exists(samples_dir):
        print(f"❌ Sample images directory not found at {samples_dir}")
        return

    samples = list(Path(samples_dir).glob("*.png"))
    if not samples:
        print("❌ No sample images found")
        return

    print(f"📷 Found {len(samples)} sample images in {samples_dir}")
    print("📷 Sample images are organized by epoch and batch, with names like:")
    print("   'epoch0_batch100_real_A.png' - Real images from domain A")
    print("   'epoch0_batch100_fake_B.png' - Generated images from A→B")
    print("   'epoch0_batch100_real_B.png' - Real images from domain B")
    print("   'epoch0_batch100_fake_A.png' - Generated images from B→A")
    print("📁 You can view these images with any image viewer")


def main():
    parser = argparse.ArgumentParser(
        description="Download and visualize CycleGAN training data from Modal"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./viz_data",
        help="Directory to save visualization data",
    )
    parser.add_argument(
        "--no-download", action="store_true", help="Skip downloading data from Modal"
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="Don't start TensorBoard server"
    )
    args = parser.parse_args()

    # Download data from Modal if requested
    if not args.no_download:
        if not download_modal_viz_data(args.output):
            print("Continuing with existing data (if any)...")

    # Display information about sample images
    display_samples_info(args.output)

    # Start TensorBoard if requested
    tensorboard_process = None
    if not args.no_tensorboard:
        tensorboard_process = start_tensorboard(args.output)
        if tensorboard_process:
            # Open browser with TensorBoard
            webbrowser.open("http://localhost:6006")

    if tensorboard_process:
        try:
            print("\nPress Ctrl+C to stop TensorBoard and exit")
            # Keep the script running as long as TensorBoard is running
            while tensorboard_process.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down TensorBoard...")
            tensorboard_process.terminate()
            print("Done!")


if __name__ == "__main__":
    main()
