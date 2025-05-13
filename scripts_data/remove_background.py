import os
import fal_client
import requests
from pathlib import Path
import time
import base64
import io
from PIL import Image

# Define directories
INPUT_DIR = "character_dataset_128"
OUTPUT_DIR = "character_dataset_128_no_bg"

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


def replace_background_with_white(img_data, size=(128, 128)):
    """
    Replace transparent background with white and resize to 128x128
    """
    # Load image from bytes
    img = Image.open(io.BytesIO(img_data))

    # Convert to RGBA if it's not already
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Create a white background image
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # Composite the image with the white background
    img_with_white_bg = Image.alpha_composite(white_bg, img)

    # Convert back to RGB (removing alpha channel)
    img_with_white_bg = img_with_white_bg.convert("RGB")

    # Resize to the specified size
    img_with_white_bg = img_with_white_bg.resize(size, Image.LANCZOS)

    # Save to bytes
    output_buffer = io.BytesIO()
    img_with_white_bg.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def process_image(image_path):
    """Process a single image to remove its background"""
    # Skip non-image files
    if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        return

    # Get relative path to maintain directory structure
    rel_path = os.path.relpath(image_path, INPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, rel_path)

    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"Skipping {rel_path} (already processed)")
        return

    try:
        print(f"Processing {rel_path}...")

        # For FAL API, we need to provide a URL or base64
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
                # Create any necessary directories
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Replace transparent background with white and resize to 128x128
                processed_img_data = replace_background_with_white(response.content)

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

    # Process images sequentially to avoid API rate limits
    for image_path in image_paths:
        process_image(image_path)

    print("Processing complete!")


if __name__ == "__main__":
    main()
