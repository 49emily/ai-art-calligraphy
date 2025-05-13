#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import subprocess
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate handwriting from character edges using pix2pix"
    )
    parser.add_argument(
        "--character",
        type=str,
        required=True,
        help="Character or string of characters to render",
    )
    parser.add_argument(
        "--font_size", type=int, default=128, help="Font size for rendering"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="edges2calligraphy_pix2pix",
        help="Name of the trained pix2pix model",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="pytorch-CycleGAN-and-pix2pix/checkpoints",
        help="Directory where model checkpoints are stored",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--temp_dir", type=str, default="./temp", help="Directory for temporary files"
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="/Library/Fonts/SourceHanSansSC-VF.otf",
        help="Path to font file",
    )
    parser.add_argument(
        "--font_weight",
        type=int,
        default=100,
        help="Font weight (100=ultra thin, 900=heavy) for variable fonts",
    )
    parser.add_argument(
        "--only_extract",
        action="store_true",
        help="Only extract strokes without applying pix2pix model",
    )
    parser.add_argument(
        "--stroke_thickness",
        type=int,
        default=1,
        help="Stroke thickness adjustment (higher values = thinner strokes)",
    )
    return parser.parse_args()


def ensure_dir(directory):
    """Make sure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def render_character(character, font_path, font_size, output_path, font_weight=100):
    """Render a character using the specified font and save it as an image.

    Args:
        character: The character to render
        font_path: Path to the font file
        font_size: Font size for rendering
        output_path: Path where the output image will be saved
        font_weight: Font weight for variable fonts (100=ultra thin, 900=heavy)
    """
    # Use fixed image size of 256x256
    img_size = 256

    # Create a new image with white background
    img = Image.new("RGB", (img_size, img_size), color="white")
    draw = ImageDraw.Draw(img)

    try:
        # Load the font
        font = ImageFont.truetype(font_path, font_size)

        # If this is a variable font, try to set the weight
        if hasattr(font, "set_variation_by_name") and "wght" in getattr(
            font, "axes", {}
        ):
            font.set_variation_by_name("wght", font_weight)
        elif hasattr(font, "set_variation_by_axes") and "wght" in getattr(
            font, "axes", {}
        ):
            font.set_variation_by_axes({"wght": font_weight})
    except IOError:
        raise Exception(f"Font file not found: {font_path}")
    except Exception as e:
        print(f"Warning: Could not set font weight, using default: {e}")
        font = ImageFont.truetype(font_path, font_size)

    # Save the image first
    img.save(output_path)

    # Draw the character with precise centering
    # Use OpenCV for exact centering since it allows for more precise pixel manipulation
    img_cv = np.array(img)

    # Create a blank image for the character only
    char_img = Image.new("RGB", (img_size, img_size), color="white")
    char_draw = ImageDraw.Draw(char_img)

    # Draw the character centered as best as possible with PIL
    try:
        # Modern Pillow method
        bbox = font.getbbox(character)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        try:
            # Older Pillow method
            text_width, text_height = font.getsize(character)
        except AttributeError:
            # Fallback to even older method
            text_width, text_height = char_draw.textsize(character, font=font)

    # Calculate position for center alignment
    position = (img_size // 2 - text_width // 2, img_size // 2 - text_height // 2)

    # Draw the character
    char_draw.text(position, character, font=font, fill="black")

    # Convert to OpenCV format
    char_img_cv = cv2.cvtColor(np.array(char_img), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(char_img_cv, cv2.COLOR_BGR2GRAY)

    # Threshold to get character pixels
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the bounding rectangle for all contours
        x_min, y_min = img_size, img_size
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Calculate character center
        char_width = x_max - x_min
        char_height = y_max - y_min
        char_center_x = x_min + char_width // 2
        char_center_y = y_min + char_height // 2

        # Calculate image center
        img_center_x = img_size // 2
        img_center_y = img_size // 2

        # Calculate shift needed to center the character precisely
        shift_x = img_center_x - char_center_x
        shift_y = img_center_y - char_center_y

        # Create a new image with the character perfectly centered
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered = cv2.warpAffine(
            char_img_cv, M, (img_size, img_size), borderValue=(255, 255, 255)
        )

        # Save the centered image
        cv2.imwrite(output_path, centered)
    else:
        # If no contours found, just save the original PIL drawing
        char_img.save(output_path)

    return output_path


def verify_center_and_adjust(image_path):
    """
    This function is deprecated and no longer used.
    The new centering logic in render_character makes this unnecessary.
    """
    pass


def extract_strokes(image_path, output_path, thickness=1):
    """Extract character strokes from the rendered image.
    Creates an image with thin black strokes on a white background.

    Args:
        image_path: Path to the input image
        output_path: Path to save the output image
        thickness: Stroke thickness adjustment (higher values = thinner strokes)
    """
    # Read image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary representation of the character
    # Using standard threshold to get black text on white background
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Thin the strokes if requested
    if thickness > 0:
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=thickness)
        result = eroded
    else:
        result = binary

    # Save the strokes image
    cv2.imwrite(output_path, result)
    return output_path


def apply_pix2pix(
    edge_image_dir, model_name, checkpoints_dir, results_dir, character, char_count=0
):
    """Apply the pix2pix model to generate handwriting from edges.

    Args:
        edge_image_dir: Directory containing the edge images
        model_name: Name of the pix2pix model
        checkpoints_dir: Directory containing model checkpoints
        results_dir: Directory to save results
        character: The character being processed
        char_count: Counter indicating how many times this character has appeared
    """
    # Path to test.py in the pix2pix repository
    test_script = "pytorch-CycleGAN-and-pix2pix/test.py"

    # Check if test script exists
    if not os.path.exists(test_script):
        raise FileNotFoundError(
            f"Could not find pix2pix test script at {test_script}. "
            "Please check the path and make sure the repository is properly set up."
        )

    # Create a character-specific results directory
    char_results_dir = os.path.join(results_dir, f"char_{character}_{char_count}")
    ensure_dir(char_results_dir)

    # Construct the command
    cmd = [
        "python",
        test_script,
        "--dataroot",
        edge_image_dir,
        "--name",
        model_name,
        "--model",
        "test",
        "--netG",
        "unet_256",
        "--direction",
        "BtoA",
        "--dataset_mode",
        "single",
        "--norm",
        "batch",
        "--checkpoints_dir",
        checkpoints_dir,
        "--results_dir",
        char_results_dir,
        "--gpu_ids",
        "-1",  # Use CPU instead of GPU
    ]

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running pix2pix model: {e}")
        print(
            "This might be due to CUDA/GPU issues. Make sure PyTorch is properly installed."
        )
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    return char_results_dir


def process_single_character(character, args, char_count=0):
    """Process a single character through the pipeline.

    Args:
        character: The character to process
        args: Command line arguments
        char_count: Counter indicating how many times this character has appeared

    Returns:
        Tuple of (stroke_image_path, output_path, result_dir)
    """
    # Create a temporary directory for the character image
    char_dir_name = f"char_input_{char_count}"
    char_temp_dir = os.path.join(args.temp_dir, char_dir_name)
    ensure_dir(char_temp_dir)

    # Render the character
    print(f"Rendering character: '{character}'")
    print(f"Using font: {args.font_path}, weight: {args.font_weight}")
    char_image_path = os.path.join(char_temp_dir, "char.png")
    render_character(
        character, args.font_path, args.font_size, char_image_path, args.font_weight
    )

    # Extract strokes
    print(
        f"Extracting character strokes (thickness adjustment: {args.stroke_thickness})"
    )
    stroke_image_path = os.path.join(char_temp_dir, "edges.png")
    extract_strokes(char_image_path, stroke_image_path, args.stroke_thickness)

    # Create character-specific results directory
    char_results_dir = os.path.join(args.results_dir, f"char_{character}_{char_count}")
    ensure_dir(char_results_dir)

    # Copy stroke image to character-specific results directory
    strokes_output_path = os.path.join(char_results_dir, f"{character}_strokes.png")
    shutil.copy(stroke_image_path, strokes_output_path)
    print(f"Stroke image saved to: {strokes_output_path}")

    # Apply pix2pix model if not only extracting strokes
    if not args.only_extract:
        print(f"Applying pix2pix model to generate handwriting for '{character}'")
        result_dir = apply_pix2pix(
            char_temp_dir,
            args.model_name,
            args.checkpoints_dir,
            args.results_dir,
            character,
            char_count,
        )
        print(f"Results for '{character}' saved to: {result_dir}")
        return stroke_image_path, strokes_output_path, result_dir

    return stroke_image_path, strokes_output_path, char_results_dir


def main():
    try:
        args = parse_args()

        # Ensure directories exist
        ensure_dir(args.temp_dir)
        ensure_dir(args.results_dir)

        # Dictionary to track character occurrences
        char_counts = {}

        # Check if input is a single character or a string
        if len(args.character) == 1:
            # Process a single character
            char = args.character
            # Initialize count for this character if it doesn't exist
            if char not in char_counts:
                char_counts[char] = 0

            stroke_path, output_path, result_dir = process_single_character(
                char, args, char_counts[char]
            )

            # Increment the count for this character
            char_counts[char] += 1

            if not args.only_extract:
                print(f"Results for '{char}' saved to: {result_dir}")
        else:
            # Process each character in the string
            print(f"Processing {len(args.character)} characters: {args.character}")

            results = []
            for char in args.character:
                # Initialize count for this character if it doesn't exist
                if char not in char_counts:
                    char_counts[char] = 0

                stroke_path, output_path, result_dir = process_single_character(
                    char, args, char_counts[char]
                )

                results.append((char, output_path, result_dir))

                # Increment the count for this character
                char_counts[char] += 1

            # Print summary of all processed characters
            print("\nProcessing complete. Summary:")
            for char, _, result_dir in results:
                print(f"Character '{char}': {result_dir}")

        if args.only_extract:
            print("Stroke extraction complete. Skipping pix2pix model application.")
        else:
            print(
                f"All results saved in individual character folders within {args.results_dir}"
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if all required files and paths exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if "CUDA" in str(e) or "cuda" in str(e):
            print(
                "\nThis appears to be a CUDA/GPU error. The script is now configured to use CPU instead,"
            )
            print(
                "but you may need to check your PyTorch installation or CUDA compatibility."
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
