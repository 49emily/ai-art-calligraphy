#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image
import numpy as np
from pathlib import Path


def get_character_image(char, count=None):
    """
    Get the image for a specific character with optional occurrence count

    Args:
        char: The character to get the image for
        count: The occurrence count of this character (0 for first occurrence, 1 for second, etc.)
    """
    # Check if this character has a specific count
    results_dir = Path("results")

    # If count is provided, use it
    print(f"Getting image for character '{char}' with count {count}")
    if count is not None:
        char_dir = results_dir / f"char_{char}_{count}"
        print(f"Checking directory: {char_dir}")
        if char_dir.exists():
            img_path = char_dir / "characters/test_latest/images/char_fake.png"
            if img_path.exists():
                return Image.open(img_path)

    # If no count or the specified count doesn't exist, find the first matching folder
    for item in results_dir.glob(f"char_{char}_*"):
        if item.is_dir():
            img_path = item / "characters/test_latest/images/char_fake.png"
            if img_path.exists():
                return Image.open(img_path)

    # If we couldn't find the character, return None
    print(f"Warning: Could not find image for character '{char}'")
    return None


def crop_image(img, crop_pixels=20):
    """
    Crop image by specified number of pixels on all sides

    Args:
        img: PIL Image object
        crop_pixels: Number of pixels to crop from each side

    Returns:
        Cropped PIL Image
    """
    width, height = img.size
    return img.crop(
        (
            crop_pixels,  # left
            crop_pixels,  # top
            width - crop_pixels,  # right
            height - crop_pixels,  # bottom
        )
    )


def create_letter(
    text,
    chars_per_line=13,
    spacing=10,
    output_path="letter.png",
    crop_pixels=20,
    padding=20,
):
    """
    Create a letter by arranging character images

    Args:
        text: The text to convert to a letter
        chars_per_line: Approximate number of characters per line
        spacing: Number of pixels between characters
        output_path: Path to save the resulting image
        crop_pixels: Number of pixels to crop from each side of character images
        padding: Padding around the entire letter in pixels
    """
    lines = text.split("\n")
    formatted_lines = []

    # Process each line to respect the chars_per_line limit
    for line in lines:
        if len(line) <= chars_per_line:
            formatted_lines.append(line)
        else:
            # Split long lines
            current_pos = 0
            while current_pos < len(line):
                end_pos = min(current_pos + chars_per_line, len(line))
                formatted_lines.append(line[current_pos:end_pos])
                current_pos = end_pos

    # Fixed dimensions for all characters after cropping
    original_size = 256
    char_width = original_size - (2 * crop_pixels)
    char_height = original_size - (2 * crop_pixels)

    # Calculate dimensions of the content area (without padding)
    max_chars_in_line = max(len(line) for line in formatted_lines)
    content_width = (char_width + spacing) * max_chars_in_line - spacing
    content_height = (char_height + spacing) * len(formatted_lines) - spacing

    # Add padding to get final image dimensions
    img_width = content_width + (padding * 2)
    img_height = content_height + (padding * 2)

    # Track character counts
    char_counts = {}

    # Create a canvas with the background color
    letter_img = Image.new("RGB", (img_width, img_height), (191, 187, 179))

    # Place character images on the canvas with padding offset
    for line_idx, line in enumerate(formatted_lines):
        for char_idx, char in enumerate(line):
            # Skip spaces
            if char == " ":
                continue

            # Initialize count for this character if it doesn't exist
            if char not in char_counts:
                char_counts[char] = 0

            # Get image for this character occurrence
            char_img = get_character_image(char, char_counts[char])

            # Increment the count for next occurrence
            char_counts[char] += 1

            if char_img:
                # Ensure image is 256x256
                if char_img.size != (original_size, original_size):
                    char_img = char_img.resize(
                        (original_size, original_size), Image.LANCZOS
                    )

                # Crop the image
                char_img = crop_image(char_img, crop_pixels)

                # Calculate position with padding offset
                x = padding + char_idx * (char_width + spacing)
                y = padding + line_idx * (char_height + spacing)

                # Paste the character onto the canvas
                letter_img.paste(char_img, (x, y))

    # Save the final image
    letter_img.save(output_path)
    print(f"Letter created and saved to {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a letter from Chinese character images"
    )
    parser.add_argument(
        "text", help="Text to convert to a letter (use \\n for line breaks)"
    )
    parser.add_argument("--output", "-o", default="letter.png", help="Output file path")
    parser.add_argument(
        "--chars-per-line",
        "-c",
        type=int,
        default=13,
        help="Approximate number of characters per line",
    )
    parser.add_argument(
        "--spacing",
        "-s",
        type=int,
        default=10,
        help="Spacing between characters in pixels",
    )
    parser.add_argument(
        "--crop",
        "-p",
        type=int,
        default=20,
        help="Number of pixels to crop from each side of character images",
    )
    parser.add_argument(
        "--padding",
        "-m",
        type=int,
        default=50,
        help="Padding around the entire letter in pixels",
    )

    args = parser.parse_args()

    # Replace literal \n with actual newlines
    text = args.text.replace("\\n", "\n")

    create_letter(
        text, args.chars_per_line, args.spacing, args.output, args.crop, args.padding
    )
