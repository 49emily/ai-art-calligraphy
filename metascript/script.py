from PIL import Image, ImageDraw, ImageFont
import os

# Path to your variable font (e.g., Simplified Chinese .otf or .ttf)
font_path = "SourceHanSansSC-VF.otf"  # or .ttf
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Choose characters to render
characters = "晖我走了我们又将暂时分离一段时间。经历了一年的感情磨合，我们彼此己很难缺少对方，我们每次的努力都是为了将来的作准备。晖的宽容与理解是我最大的动力与安慰。这半年的努力我相信会对我们的未来起到很大的作用。虽然不在你身边，但我时刻会想念你，每分每秒。爱你的健"  # Add more as needed

# Set image and font size
img_size = (256, 256)
font_size = 180

# Load font (Pillow supports variable OTF/TTF)
font = ImageFont.truetype(font_path, font_size)

# Set weight axis directly - check if this method is available
# For newer versions of Pillow
try:
    # 700-900 is bold/heavy range in most fonts
    font.set_variation_by_axes([("wght", 600)])
except (AttributeError, TypeError):
    # For older versions or different font configurations
    # Note that not all variable fonts support named variations
    try:
        font.set_variation_by_name("Normal")
    except (AttributeError, OSError):
        print("Could not set font weight - using default weight")

# Render each character
for ch in characters:
    img = Image.new("L", img_size, color=255)  # white background
    draw = ImageDraw.Draw(img)

    # Calculate the true center position
    center_x, center_y = img_size[0] // 2, img_size[1] // 2

    # Get text bounding box at the origin
    left, top, right, bottom = draw.textbbox((0, 0), ch, font=font)

    # Calculate the position to center the text properly
    # Adjust the origin by the left and top offsets
    position = (
        center_x - (right - left) // 2 - left,
        center_y - (bottom - top) // 2 - top,
    )

    # Draw the text
    draw.text(position, ch, fill=0, font=font)

    img.save(os.path.join(output_dir, f"{ord(ch)}.png"))
