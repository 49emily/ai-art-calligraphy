#!/usr/bin/env python3
import os
import sys
import argparse
import re
from PIL import Image
import numpy as np
import datetime


def create_character_grid(input_folder, output_file):
    """
    Creates a grid image from all character images in the input folder.
    Each row contains 10 characters, and the grid is as long as needed.

    Args:
        input_folder: Folder containing 128x128 character images
        output_file: Path to save the output grid image
    """
    # Get all image files in the folder
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(input_folder, file))

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    # Sort the files numerically based on the file name
    def natural_sort_key(s):
        # Extract the base filename without extension
        base_name = os.path.splitext(os.path.basename(s))[0]

        # Try to convert the entire filename to an integer for simple numeric filenames (0.png, 1.png, etc.)
        try:
            return int(base_name)
        except ValueError:
            # If that fails, extract any numbers from the filename
            numbers = re.findall(r"\d+", base_name)
            if numbers:
                # Return the first number found
                return int(numbers[0])
            # If no numbers, return the original string for alphabetical sort
            return base_name

    image_files.sort(key=natural_sort_key)
    print(f"Processing {len(image_files)} images in numerical order")

    # Calculate grid dimensions
    num_images = len(image_files)
    images_per_row = 10
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Ceiling division

    # Create a blank canvas
    canvas_width = images_per_row * 128
    canvas_height = num_rows * 128
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    # Place images on the canvas
    for i, image_file in enumerate(image_files):
        try:
            img = Image.open(image_file)

            # Verify image size
            if img.size != (128, 128):
                print(f"Warning: Image {image_file} is not 128x128. Resizing.")
                img = img.resize((128, 128))

            # Calculate position
            row = i // images_per_row
            col = i % images_per_row
            x = col * 128
            y = row * 128

            # Paste onto canvas
            canvas.paste(img, (x, y))

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Save the result
    canvas.save(output_file)
    print(f"Created grid image with {num_images} characters at {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create a grid of character images")
    parser.add_argument(
        "input_folder", help="Folder containing 128x128 character images"
    )

    # Generate default filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"character_grid_{timestamp}.png"

    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help=f"Output image file name (default: {default_output})",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        return 1

    create_character_grid(args.input_folder, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
