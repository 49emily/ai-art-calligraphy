#!/usr/bin/python3

import modal
import os
import sys
from pathlib import Path

# Define the Modal App
app = modal.App("cyclegan-inference")
image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "pillow", "numpy", "tqdm"
)

# Use a smaller GPU for inference
gpu = modal.gpu.T4()

# Reference the same models volume that training uses
models_volume = modal.Volume.from_name("cyclegan-models")


@app.function(
    image=image,
    gpu=gpu,
    mounts=[modal.Mount.from_local_dir("./", remote_path="/root/cyclegan/")],
    volumes={"/root/cyclegan/model_output": models_volume},
)
def run_inference(
    input_path,
    output_path,
    model_name="final_netG_A2B.pth",
    direction="A2B",
    size=256,
    use_local_model=False,
):
    """Run CycleGAN inference on Modal"""
    import sys
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np

    os.chdir("/root/cyclegan")
    sys.path.append("/root/cyclegan")

    # Import project files - handle both import methods
    try:
        from models import Generator
    except ModuleNotFoundError:
        # Try with explicit import path
        sys.path.append("/root/cyclegan")
        from cyclegan.models import Generator

    # Use modal_utils instead of utils
    from modal_utils import weights_init_normal

    # Always save the output to the Modal volume
    output_vol_path = f"/root/cyclegan/model_output/{os.path.basename(output_path)}"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_vol_path), exist_ok=True)

    # Determine which generator to use based on direction
    if direction == "A2B":
        netG = Generator(3, 3)  # From domain A to domain B
    else:  # "B2A"
        netG = Generator(3, 3)  # From domain B to domain A

    # Determine the model path
    if use_local_model:
        # Use local model path provided directly
        model_path = model_name
    else:
        # Use model from Modal volume
        model_path = f"/root/cyclegan/model_output/{model_name}"

    print(f"Loading model from: {model_path}")

    # Load the model
    netG.load_state_dict(torch.load(model_path))
    netG.to(torch.device("cuda"))
    netG.eval()  # Set to evaluation mode

    # Preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load and preprocess the input image
    input_image = Image.open(input_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(torch.device("cuda"))

    # Generate the output image
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    # Postprocess the output tensor to an image
    output_tensor = output_tensor.cpu().squeeze(0)
    output_tensor = (output_tensor + 1) / 2  # Normalize from [-1, 1] to [0, 1]
    output_image = transforms.ToPILImage()(output_tensor)

    # Save the output image to both the volume and the local output path if specified
    output_image.save(output_vol_path)
    print(f"Transformed image saved to Modal volume at: {output_vol_path}")

    # Also save locally if specified path is different from volume path
    if output_path != output_vol_path:
        # Make sure the local directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_image.save(output_path)
        print(f"Transformed image also saved locally to: {output_path}")

    return f"Inference completed successfully. Output saved to Modal volume: {os.path.basename(output_vol_path)}"


@app.function(
    image=image,
    gpu=gpu,
    mounts=[modal.Mount.from_local_dir("./", remote_path="/root/cyclegan/")],
    volumes={"/root/cyclegan/model_output": models_volume},
)
def run_batch_inference(
    input_dir,
    output_dir,
    model_name="final_netG_A2B.pth",
    direction="A2B",
    size=256,
    use_local_models=False,
):
    """Run CycleGAN inference on a folder of images"""
    import sys
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import glob
    from tqdm import tqdm

    os.chdir("/root/cyclegan")
    sys.path.append("/root/cyclegan")

    # Import project files
    try:
        from models import Generator
    except ModuleNotFoundError:
        sys.path.append("/root/cyclegan")
        from cyclegan.models import Generator

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define Modal volume output directory path
    modal_volume_output_dir = os.path.join("/root/cyclegan/model_output", output_dir)
    os.makedirs(modal_volume_output_dir, exist_ok=True)

    # Initialize model based on direction
    netG = Generator(3, 3)

    # Determine model path
    if use_local_models:
        model_path = model_name
    else:
        model_path = f"/root/cyclegan/model_output/{model_name}"

    # Load model
    print(f"Loading {direction} model from: {model_path}")
    netG.load_state_dict(torch.load(model_path))
    netG.to(torch.device("cuda"))
    netG.eval()

    # Preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Get list of images from input directory
    image_paths = (
        glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.jpeg"))
        + glob.glob(os.path.join(input_dir, "*.png"))
    )

    print(f"Found {len(image_paths)} images to process")

    # Process each image
    results = []
    for i, img_path in enumerate(tqdm(image_paths)):
        filename = os.path.basename(img_path)
        base_filename = os.path.splitext(filename)[0]

        try:
            # Load and preprocess input image
            input_image = Image.open(img_path).convert("RGB")
            input_tensor = transform(input_image).unsqueeze(0).to(torch.device("cuda"))

            # Generate output
            with torch.no_grad():
                output = netG(input_tensor)

            # Save result
            output = output.cpu().squeeze(0)
            output = (output + 1) / 2  # Normalize to [0, 1]
            output_image = transforms.ToPILImage()(output)

            # Save to Modal volume
            vol_out_path = os.path.join(modal_volume_output_dir, f"{base_filename}.png")
            output_image.save(vol_out_path)

            # Save to specified output directory (which may be local when downloaded)
            out_path = os.path.join(output_dir, f"{base_filename}.png")
            output_image.save(out_path)

            results.append(f"Successfully processed {filename}")

        except Exception as e:
            results.append(f"Error processing {filename}: {str(e)}")

    return {
        "total_images": len(image_paths),
        "results": results,
        "output_dir": output_dir,
        "modal_volume_dir": modal_volume_output_dir,
    }


@app.function(volumes={"/root/model_output": models_volume})
def list_available_models():
    """List all available models in the Modal volume"""
    models = os.listdir("/root/model_output")
    return [model for model in models if model.endswith(".pth")]


@app.local_entrypoint()
def main():
    # Define your parameters here
    input_dir = "遇见你是我最幸运的意"  # Directory containing input images
    output_dir = "run_20250507_051848/遇见你是我最幸运的意"  # Directory to save results
    model_name = "run_20250507_051848/final_netG_A2B.pth"  # Model to use for inference
    direction = "A2B"  # Direction of transformation: "A2B" or "B2A"
    size = 128  # Size to resize images before processing
    use_local_models = False  # Use models from Modal volume by default

    # Uncomment to list available models
    # models = list_available_models.remote()
    # print("Available models in Modal volume:")
    # for model in models:
    #     print(f"  - {model}")

    # Run batch inference
    result = run_batch_inference.remote(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=model_name,
        direction=direction,
        size=size,
        use_local_models=use_local_models,
    )
    print(f"Processed {result['total_images']} images")
    print(f"Results saved to: {result['output_dir']}")
    print(f"Results also saved to Modal volume: {result['modal_volume_dir']}")
