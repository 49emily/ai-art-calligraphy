"""
Modal Data Uploader for CycleGAN and pix2pix datasets

This script helps prepare and upload datasets to Modal for training.
It can package datasets locally and upload them to Modal's persistent storage.
"""

import os
import sys
import argparse
import shutil
import time
import modal
from modal import Volume


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare and upload datasets to Modal")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., facades, maps, edges2shoes)",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--volume_name",
        type=str,
        default=None,
        help='Name for the Modal volume (defaults to "cyclegan-{dataset_name}")',
    )
    parser.add_argument(
        "--compress", action="store_true", help="Compress the dataset before uploading"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing volume if it exists",
    )
    return parser.parse_args()


def compress_dataset(dataset_path, output_path):
    """Compress the dataset directory into a zip file."""
    import zipfile

    print(f"Compressing dataset at {dataset_path} to {output_path}...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(dataset_path))
                zipf.write(file_path, arcname)

    return output_path


def create_volume_and_upload(args):
    """Create a Modal volume and upload the dataset."""
    dataset_name = args.dataset_name
    volume_name = args.volume_name or f"cyclegan-{dataset_name}"

    # Check if volume exists
    try:
        existing_volumes = modal.Volume.list()
        if volume_name in existing_volumes and not args.overwrite:
            print(
                f"Volume '{volume_name}' already exists. Use --overwrite to replace it."
            )
            return False
    except Exception as e:
        print(f"Error checking existing volumes: {e}")
        return False

    # Create volume
    try:
        volume = modal.Volume.create(volume_name)
        print(f"Created Modal volume: {volume_name}")
    except Exception as e:
        print(f"Error creating volume: {e}")
        return False

    # Copy dataset to a temporary directory
    temp_dir = "/tmp/modal-dataset"
    os.makedirs(temp_dir, exist_ok=True)

    if args.compress:
        # Compress the dataset
        zip_path = os.path.join(temp_dir, f"{dataset_name}.zip")
        compress_dataset(args.dataset_path, zip_path)

        # Upload zip file
        with volume.get_client() as client:
            client.put(zip_path, f"/{dataset_name}.zip")
            print(f"Uploaded compressed dataset to /{dataset_name}.zip")
    else:
        # Create dataset directory structure
        dataset_dir = os.path.join(temp_dir, dataset_name)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)

        # Copy dataset files
        shutil.copytree(args.dataset_path, dataset_dir)
        print(f"Copied dataset to temporary directory: {dataset_dir}")

        # Upload dataset directory
        with volume.get_client() as client:
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, temp_dir)
                    remote_path = f"/{relative_path}"

                    # Create directories if needed
                    remote_dir = os.path.dirname(remote_path)
                    client.makedirs(remote_dir)

                    # Upload file
                    client.put(local_path, remote_path)
                    print(f"Uploaded: {remote_path}")

    print(f"\nDataset successfully uploaded to Modal volume: {volume_name}")
    print(f"Use this volume in your Modal script with:")
    print(f"    volume = modal.Volume.from_name('{volume_name}')")
    print(f"    stub.function(..., volumes=[(volume, '/datasets/{dataset_name}')])")

    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    return True


def main():
    """Main function."""
    args = parse_args()

    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist.")
        return 1

    # Create volume and upload dataset
    success = create_volume_and_upload(args)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
