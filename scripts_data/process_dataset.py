import os
import fal_client
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import base64
import tempfile
import urllib.parse
from PIL import Image
import io

# Define directories
INPUT_DIR = "character_dataset_upscaled"
OUTPUT_DIR = "character_dataset_upscaled_no_bg"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])


def encode_image_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def resize_to_512px_square_with_transparency(img_data):
    """
    Resize image so that the larger side is 512px and
    extend the other side to 512px with transparency
    """
    # Load image from bytes
    img = Image.open(io.BytesIO(img_data))

    # Convert to RGBA if it's not already
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Get dimensions
    width, height = img.size

    # Calculate new dimensions while preserving aspect ratio
    if width > height:
        # Width is the larger dimension
        new_width = 512
        new_height = int((height / width) * 512)
    else:
        # Height is the larger dimension
        new_height = 512
        new_width = int((width / height) * 512)

    # Resize image while preserving aspect ratio
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create a new transparent image of 512x512
    new_img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

    # Paste resized image at the center
    paste_x = (512 - new_width) // 2
    paste_y = (512 - new_height) // 2
    new_img.paste(img, (paste_x, paste_y), img)

    # Save to bytes
    output_buffer = io.BytesIO()
    new_img.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def process_image(image_path):
    """Process a single image to remove its background"""
    # Skip non-image files
    if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        return

    # Get relative path to maintain directory structure
    rel_path = os.path.relpath(image_path, INPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, rel_path)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"Skipping {rel_path} (already processed)")
        return

    try:
        print(f"Processing {rel_path}...")

        # For FAL API, we need to provide a URL or base64
        # We'll use base64 encoding for simplicity
        image_base64 = encode_image_base64(image_path)
        image_data_uri = f"data:image/png;base64,{image_base64}"

        # Use fal-ai's background removal endpoint
        result = fal_client.subscribe(
            "fal-ai/bria/background/remove",
            arguments={"image_url": image_data_uri},
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        # Save the result
        if (
            result
            and "image" in result
            and isinstance(result["image"], dict)
            and "url" in result["image"]
        ):
            # Download the result
            img_url = result["image"]["url"]
            response = requests.get(img_url)
            if response.status_code == 200:
                # Resize the image to 512x512 with transparency
                processed_img_data = resize_to_512px_square_with_transparency(
                    response.content
                )

                # Save the processed image
                with open(output_path, "wb") as f:
                    f.write(processed_img_data)
                print(f"Saved processed image to {output_path}")
            else:
                print(f"Failed to download result for {rel_path}")
        else:
            print(f"Failed to process {rel_path} - unexpected response format")
            print(f"Response: {result}")

    except Exception as e:
        print(f"Error processing {rel_path}: {str(e)}")

    # Add a small delay to avoid overwhelming the API
    time.sleep(0.5)


def main():
    # Find all images in dataset directory
    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")) and not file.startswith(
                "."
            ):
                image_paths.append(os.path.join(root, file))

    print(f"Found {len(image_paths)} images to process")

    # Process images in parallel (adjust max_workers as needed)
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_image, image_paths)

    print("Processing complete!")


if __name__ == "__main__":
    main()
