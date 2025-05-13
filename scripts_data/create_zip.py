import os
import re
import shutil
from PIL import Image


def extract_character(filename):
    # Pattern matches either {character}.png or {character}_{i}.png
    match = re.match(r"(.)(?:_\d+)?\.png$", filename)
    if match:
        return match.group(1)
    return None


def create_text_files():
    source_directory = "character_dataset_upscaled_no_bg"
    target_directory = "kolors_training_data"

    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Source directory {source_directory} does not exist.")
        return

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created directory {target_directory}")

    # Get all PNG files in the source directory
    png_files = [f for f in os.listdir(source_directory) if f.endswith(".png")]

    for png_file in png_files:
        # Extract character from filename
        character = extract_character(png_file)

        if character:
            # Process the image - make transparent pixels white
            source_path = os.path.join(source_directory, png_file)
            target_path = os.path.join(target_directory, png_file)

            # Open image and convert transparent pixels to white
            img = Image.open(source_path)
            if img.mode == "RGBA":
                # Create a white background image
                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Paste the image on the white background using the alpha channel as mask
                white_bg.paste(img, (0, 0), img)
                # Convert to RGB (no transparency)
                white_bg = white_bg.convert("RGB")
                # Save the new image
                white_bg.save(target_path)
                print(f"Copied and made transparent pixels white in {png_file}")
            else:
                # If the image doesn't have transparency, just copy it
                shutil.copy2(source_path, target_path)
                print(f"Copied {png_file} to {target_directory}")

            # Create text filename
            txt_filename = os.path.splitext(png_file)[0] + ".txt"
            txt_path = os.path.join(target_directory, txt_filename)

            # Write the character to the text file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"zsh手写的“{character}”字")

            print(f"Created {txt_path} with character '{character}'")
        else:
            print(f"Could not extract character from {png_file}")


if __name__ == "__main__":
    create_text_files()
