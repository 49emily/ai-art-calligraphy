#!/usr/bin/env python3
import sys
import os
import random
from PIL import Image, ImageDraw, ImageFont

# Define a list of common Chinese characters
COMMON_CHINESE_CHARS = [
    # Basic greetings and everyday words
    "你",
    "好",
    "我",
    "是",
    "的",
    "了",
    "在",
    "有",
    "和",
    "人",
    "这",
    "一",
    "不",
    "来",
    "到",
    "大",
    "中",
    "上",
    "下",
    "小",
    "水",
    "火",
    "山",
    "口",
    "日",
    "月",
    "木",
    "田",
    "土",
    "年",
    "家",
    "学",
    "生",
    "工",
    "心",
    "出",
    "子",
    "门",
    "手",
    "文",
    "天",
    "开",
    "道",
    "时",
    "气",
    "国",
    "说",
    "去",
    "会",
    "东",
    "西",
    "南",
    "北",
    "前",
    "后",
    "里",
    "外",
    "高",
    "低",
    "中",
    "长",
    "见",
    "行",
    "很",
    "老",
    "少",
    "多",
    "少",
    "为",
    "什",
    "么",
    "图",
    "书",
    "车",
    "马",
    "鱼",
    "羊",
    "鸟",
    "飞",
    "走",
    "入",
    "回",
    "想",
    "听",
    "看",
    "言",
    "语",
    "爱",
    "吃",
    "喝",
]


def get_random_chinese_char():
    """Get a random character from the common Chinese characters list."""
    return random.choice(COMMON_CHINESE_CHARS)


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


def generate_characters_from_text(text, output_dir="test", font_path="simhei.ttf"):
    """
    Generate ground truth images for each character in the provided text string.

    Args:
        text: A string of characters to generate images for
        output_dir: Directory to save the images (default: "test")
        font_path: Path to the font file to use (default: "simhei.ttf")

    Returns:
        A list of paths to the generated images
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if not os.path.exists(font_path):
        print(f"Warning: Font file '{font_path}' not found. The script may fail.")

    output_paths = []

    # Generate an image for each character in the text
    for i, character in enumerate(text):
        if character.isspace():
            continue  # Skip spaces

        output_path = os.path.join(output_dir, f"{i}.png")
        print(f"Generating image for character: {character}")
        create_ground_truth_image(character, output_path, font_path)
        output_paths.append(output_path)

    print(f"{len(output_paths)} character images saved to {output_dir}/ directory")
    return output_paths


def main():
    # Check if the user wants to generate from text
    if len(sys.argv) > 1 and sys.argv[1] == "--from-text":
        if len(sys.argv) < 3:
            print("Error: Please provide text after the --from-text flag")
            sys.exit(1)

        text = sys.argv[2]
        font_path = sys.argv[3] if len(sys.argv) > 3 else "simhei.ttf"
        generate_characters_from_text(
            text, text.replace(" ", "").replace("\n", "")[:10], font_path
        )
        return

    # Generate 50 random characters if none provided
    if len(sys.argv) < 2:
        print("No character provided. Generating 50 random Chinese characters.")

        # Create test directory if it doesn't exist
        test_dir = "test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            print(f"Created directory: {test_dir}")

        font_path = sys.argv[1] if len(sys.argv) > 1 else "simhei.ttf"

        if not os.path.exists(font_path):
            print(f"Warning: Font file '{font_path}' not found. The script may fail.")

        # Generate 50 random characters
        for i in range(50):
            character = get_random_chinese_char()
            output_path = os.path.join(test_dir, f"char_{i:02d}_{character}.png")
            print(f"Generating image for character: {character}")
            create_ground_truth_image(character, output_path, font_path)

        print(f"50 character images saved to {test_dir}/ directory")
        return

    character = sys.argv[1]

    if len(character) != 1:
        print("Error: Please provide exactly one character")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{character}.png"
    font_path = sys.argv[3] if len(sys.argv) > 3 else "simhei.ttf"

    if not os.path.exists(font_path):
        print(f"Warning: Font file '{font_path}' not found. The script may fail.")

    print(f"Generating image for character: {character}")
    print(f"Output path: {output_path}")
    print(f"Font path: {font_path}")

    create_ground_truth_image(character, output_path, font_path)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    main()
