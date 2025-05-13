#!/usr/bin/env python3
"""
Helper script for uploading data to Modal volumes.
Usage:
  python upload_data.py --source_dir ./my_images --volume_name dreambooth-data --volume_path dog
"""

import os
import argparse
import modal
from tqdm import tqdm
from PIL import Image
import io


def parse_args():
    parser = argparse.ArgumentParser(description="Upload data to Modal volume")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Local directory with source files",
    )
    parser.add_argument(
        "--volume_name", type=str, required=True, help="Name of the Modal volume"
    )
    parser.add_argument(
        "--volume_path",
        type=str,
        required=True,
        help="Path inside the volume to upload files",
    )
    parser.add_argument(
        "--resize", type=int, default=None, help="Resize images to this size (optional)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Make sure the source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist")
        return

    # Connect to the Modal volume
    volume = modal.Volume.from_name(args.volume_name)

    # Get all files from the source directory
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    files = []

    for root, _, filenames in os.walk(args.source_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                files.append(os.path.join(root, filename))

    if not files:
        print(f"No image files found in {args.source_dir}")
        return

    print(f"Found {len(files)} image files to upload")

    # Upload files with progress bar
    for file_path in tqdm(files, desc="Uploading files"):
        # Determine destination path
        rel_path = os.path.relpath(file_path, args.source_dir)
        dest_path = os.path.join(args.volume_path, rel_path)

        # Process image if resize is specified
        if args.resize:
            try:
                # Open and resize the image
                img = Image.open(file_path)

                # Preserve aspect ratio
                width, height = img.size
                if width > height:
                    new_width = args.resize
                    new_height = int(height * args.resize / width)
                else:
                    new_height = args.resize
                    new_width = int(width * args.resize / height)

                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_data = buffer.getvalue()

                # Write to volume
                volume.write(dest_path, img_data)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            # Upload file directly
            try:
                with open(file_path, "rb") as f:
                    volume.write(dest_path, f.read())
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")

    print(f"Successfully uploaded images to {args.volume_name}:{args.volume_path}")


if __name__ == "__main__":
    main()
