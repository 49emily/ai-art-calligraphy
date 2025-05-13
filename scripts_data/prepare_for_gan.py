import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re


def process_image(input_path, output_path):
    # Read the image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has an alpha channel
    has_alpha = img.shape[2] == 4 if len(img.shape) > 2 else False

    # Get the current dimensions
    h, w = img.shape[:2]

    # Determine the scaling factor to make the longer side 128px
    scale = 128 / max(h, w)

    # Calculate new dimensions while preserving aspect ratio
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize the image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a 128x128 canvas
    if has_alpha:
        # For images with alpha channel
        canvas = np.zeros((128, 128, 4), dtype=np.uint8)
    else:
        # For images without alpha channel
        canvas = np.zeros((128, 128, 3), dtype=np.uint8)

        # Sample the background color from the corners of the image
        # This assumes the corners have the background color
        corners = [resized[0, 0], resized[0, -1], resized[-1, 0], resized[-1, -1]]
        bg_color = np.median(corners, axis=0).astype(np.uint8)

        # Fill the canvas with the background color
        canvas[:] = bg_color

    # Calculate position to place the resized image (center)
    y_offset = (128 - new_h) // 2
    x_offset = (128 - new_w) // 2

    # Place the resized image on the canvas
    if has_alpha:
        # For images with alpha channel
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    else:
        # For images without alpha channel
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    # Save the processed image
    cv2.imwrite(output_path, canvas)


def extract_character_from_filename(filename):
    # Extract Chinese character from filename
    # Format can be "{char}.png" or "{char}_{number}.png"
    match = re.match(r"(.+?)(?:_\d+)?\.png", filename)
    if match:
        return match.group(1)
    return None


def create_ground_truth_image(character, output_path, font_path="simhei.ttf"):
    # Create a 128x128 white background image
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load font with increasing sizes until optimal
    font_size = 80  # Start with a reasonable size
    font = ImageFont.truetype(font_path, font_size)

    # Get text size (compatible with newer Pillow versions)
    try:
        # New method in Pillow (>=8.0.0)
        bbox = font.getbbox(character)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(character, font=font)

    # Adjust font size if needed
    while (text_width > 110 or text_height > 110) and font_size > 30:
        font_size -= 5
        font = ImageFont.truetype(font_path, font_size)
        try:
            # New method in Pillow (>=8.0.0)
            bbox = font.getbbox(character)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older Pillow versions
            text_width, text_height = draw.textsize(character, font=font)

    # Calculate position to center the text
    x = (128 - text_width) / 2
    y = (128 - text_height) / 2

    # Draw the character (compatible with newer Pillow versions)
    try:
        # New method in Pillow (>=8.0.0)
        draw.text((x, y), character, fill=(0, 0, 0), font=font)
    except TypeError:
        # Fallback for older Pillow versions
        draw.text((int(x), int(y)), character, fill=(0, 0, 0), font=font)

    # Save the image
    img.save(output_path)


def main():
    input_dir = "character_dataset"
    output_dir = "character_dataset_128"
    ground_truth_dir = "character_ground_truth"

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            ground_truth_path = os.path.join(ground_truth_dir, filename)

            # Process the input image
            process_image(input_path, output_path)

            # Create ground truth image with SimHei font
            character = extract_character_from_filename(filename)
            if character:
                try:
                    create_ground_truth_image(character, ground_truth_path)
                    print(
                        f"Processed: {filename} (created ground truth image for '{character}')"
                    )
                except Exception as e:
                    print(f"Error creating ground truth for {filename}: {e}")
            else:
                print(
                    f"Processed: {filename} (could not extract character for ground truth)"
                )


if __name__ == "__main__":
    main()
