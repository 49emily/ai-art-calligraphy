import os
import fal_client
import time
from PIL import Image
import io
import requests
import base64
from tqdm import tqdm

# Define paths
input_dir = "character_dataset"
output_dir = "character_dataset_upscaled"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])


def upload_image_to_temp_hosting(image_path):
    """
    Upload image to a temporary file hosting service and return the URL.
    Here we use imgbb.com as an example.
    You need to replace API_KEY with an actual API key from imgbb.com.
    """
    try:
        # Read the image file
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Send to FAL directly as base64
        return f"data:image/png;base64,{image_data}"
    except Exception as e:
        print(f"Error uploading image {image_path}: {e}")
        return None


def upscale_image(image_path, output_path):
    """
    Upscale an image using FAL.ai API
    """
    try:
        # Get image URL or base64
        image_data = upload_image_to_temp_hosting(image_path)
        if not image_data:
            print(f"Skipping {image_path} due to upload failure")
            return False

        # Call the FAL.ai API
        result = fal_client.subscribe(
            "fal-ai/recraft-crisp-upscale",
            arguments={"image_url": image_data},
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        # Print result structure for debugging
        print(f"Result type: {type(result)}")
        print(
            f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}"
        )

        # Save the result
        if isinstance(result, dict):
            if "image_url" in result:
                # If the result contains an image URL
                response = requests.get(result["image_url"])
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                else:
                    print(f"Failed to download upscaled image for {image_path}")
                    return False
            elif "image" in result:
                # If the result contains an image in some format
                image_content = result["image"]
                if isinstance(image_content, str) and image_content.startswith(
                    "data:image"
                ):
                    # Handle base64 encoded image
                    image_data = base64.b64decode(image_content.split(",")[1])
                    with open(output_path, "wb") as f:
                        f.write(image_data)
                elif isinstance(image_content, dict) and "url" in image_content:
                    # Handle image as an object with url
                    response = requests.get(image_content["url"])
                    if response.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                    else:
                        print(f"Failed to download upscaled image for {image_path}")
                        return False
                else:
                    print(f"Unsupported image format in result for {image_path}")
                    print(f"Image content: {image_content}")
                    return False
            else:
                print(f"No recognized image data in result for {image_path}: {result}")
                return False
        else:
            print(f"Unexpected result type for {image_path}: {type(result)}")
            print(f"Result: {result}")
            return False

        return True
    except Exception as e:
        print(f"Error upscaling {image_path}: {e}")
        return False


def process_dataset():
    """
    Process all images in the dataset
    """
    # Get all image files
    image_files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Found {len(image_files)} images to upscale")

    # Process each image
    success_count = 0
    for image_file in tqdm(image_files, desc="Upscaling images"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        print(f"Processing: {input_path}")
        # Upscale and save the image
        if upscale_image(input_path, output_path):
            success_count += 1
            # Add a small delay to avoid overwhelming the API
            time.sleep(0.5)

        # Debug with first file only
        if success_count == 0:
            print("First file failed, stopping for debugging")
            break

    print(
        f"Upscaling complete. Successfully upscaled {success_count}/{len(image_files)} images."
    )
    print(f"Upscaled images saved to {output_dir}")


if __name__ == "__main__":
    process_dataset()
