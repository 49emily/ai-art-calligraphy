import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
import json
from collections import Counter

# Create output directory if it doesn't exist
output_dir = "character_dataset"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Original and new image dimensions
original_width, original_height = 720, 1280
new_width, new_height = 1440, 2560

# Calculate scaling factors
width_scale = new_width / original_width
height_scale = new_height / original_height

# Parse XML annotations
tree = ET.parse("annotations 3.xml")
root = tree.getroot()

# Load the image
image = Image.open("characters.png")

# Create a mapping dictionary (character to Unicode code point)
char_to_code = {}
code_to_char = {}

# Keep track of character counts
char_counter = Counter()

# Process each character annotation
for i, box in enumerate(root.findall(".//box")):
    # Get character text
    text_attr = box.find(".//attribute[@name='text']")
    if text_attr is None:
        character = f"unknown_{i}"
        char_code = 1000000 + i  # Use a large number for unknown characters
    else:
        character = text_attr.text
        char_code = ord(character)  # Get Unicode code point

    # Add to mapping dictionaries
    char_to_code[character] = char_code
    code_to_char[str(char_code)] = character

    # Get bounding box coordinates
    xtl = float(box.get("xtl"))
    ytl = float(box.get("ytl"))
    xbr = float(box.get("xbr"))
    ybr = float(box.get("ybr"))

    # Scale coordinates for the new image size
    xtl = xtl * width_scale
    ytl = ytl * height_scale
    xbr = xbr * width_scale
    ybr = ybr * height_scale

    # Convert to integers for cropping
    left = int(xtl)
    top = int(ytl)
    right = int(xbr)
    bottom = int(ybr)

    # Crop the image
    cropped_img = image.crop((left, top, right, bottom))

    # Update character counter and create filename
    char_counter[character] += 1

    if char_counter[character] > 1:
        filename = f"{character}_{char_counter[character]}.png"
    else:
        filename = f"{character}.png"

    # Save the cropped image
    output_path = os.path.join(output_dir, filename)
    cropped_img.save(output_path)
    print(f"Saved {output_path} (Character: {character}, Unicode: {char_code})")

# Save the character-to-code mapping to a JSON file
mapping_file = os.path.join(output_dir, "char_mapping.json")
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(
        {"char_to_code": char_to_code, "code_to_char": code_to_char},
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Completed processing. Character images saved to '{output_dir}' directory.")
print(f"Character to Unicode mapping saved to '{mapping_file}'.")
