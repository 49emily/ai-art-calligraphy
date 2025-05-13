import os
import modal

# Define a Modal image with the necessary dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "torch>=2.1.0",
            "accelerate>=0.20.0",
            "transformers>=4.30.0",
            "diffusers>=0.21.0",
            "bitsandbytes>=0.44.0",
            "xformers",
            "safetensors",
            "datasets",
            "deepspeed",
            "pillow>=10.0.0",
            "tqdm>=4.65.0",
            "huggingface_hub>=0.18.0",
            "sentencepiece",
        ]
    )
    .apt_install("nvidia-cuda-toolkit")
)

# Create a Modal volume to store the model and training data
model_volume = modal.Volume.from_name("dreambooth-models", create_if_missing=True)
data_volume = modal.Volume.from_name("dreambooth-data", create_if_missing=True)

# Create a local directory mount to access local scripts
local_mount = modal.Mount.from_local_dir(".", remote_path="/app")

# Define the app
app = modal.App("dreambooth-training")


@app.function(
    image=image,
    gpu="a100",
    timeout=10800,  # 3 hours
    volumes={
        "/models": model_volume,
        "/data": data_volume,
    },
    mounts=[local_mount],
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_dreambooth(
    model_name: str,
    instance_dir: str,
    output_dir: str,
    instance_prompt: str,
    class_dir: str = None,
    class_prompt: str = None,
    resolution: int = 1024,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    text_encoder_lr: float = 5e-5,
    rank: int = 4,
    max_train_steps: int = 1000,
    checkpointing_steps: int = 200,
    num_class_images: int = 100,
    seed: int = 19980818,
    with_prior_preservation: bool = False,
    prior_loss_weight: float = 0.7,
    train_text_encoder: bool = True,
):
    """
    Run Dreambooth LoRA fine-tuning on Modal.

    Args:
        model_name: Path to the base model
        instance_dir: Directory containing instance images
        output_dir: Directory to save the fine-tuned model
        instance_prompt: Prompt for instance images
        class_dir: Directory containing class images (optional)
        class_prompt: Prompt for class images (optional)
        resolution: Image resolution
        train_batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        text_encoder_lr: Text encoder learning rate
        rank: LoRA rank
        max_train_steps: Maximum number of training steps
        checkpointing_steps: Number of steps between checkpoints
        num_class_images: Number of class images to generate
        seed: Random seed
        with_prior_preservation: Whether to use prior preservation
        prior_loss_weight: Weight of prior preservation loss
        train_text_encoder: Whether to train the text encoder
    """
    # Create accelerate config file
    config_content = """
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'fp16'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
    """

    with open("/tmp/accelerate_config.yaml", "w") as f:
        f.write(config_content)

    # # Create full paths on Modal volumes
    # model_path = os.path.join("/models", model_name)

    # Use paths from the local mount for training data
    instance_data_path = os.path.join("/app", instance_dir)
    class_data_path = os.path.join("/app", class_dir) if class_dir else None

    output_path = os.path.join("/models", output_dir)

    # Path to the local train_dreambooth_lora.py script
    script_path = "/app/train_dreambooth_lora.py"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Build the command
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "/app/default_config.yaml",
        script_path,
        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={instance_data_path}",
        f"--output_dir={output_path}",
        f"--instance_prompt={instance_prompt}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--text_encoder_lr={text_encoder_lr}",
        f"--lr_scheduler=polynomial",
        f"--lr_warmup_steps=100",
        f"--rank={rank}",
        f"--resolution={resolution}",
        f"--max_train_steps={max_train_steps}",
        f"--checkpointing_steps={checkpointing_steps}",
        "--center_crop",
        "--mixed_precision=fp16",
        f"--seed={seed}",
        "--img_repeat_nums=1",
        "--sample_batch_size=2",
        "--gradient_checkpointing",
        "--adam_weight_decay=1e-02",
    ]

    # Add class-related arguments only if class_dir and class_prompt are provided
    if class_dir and class_prompt and with_prior_preservation:
        cmd.append(f"--class_data_dir={class_data_path}")
        cmd.append(f"--class_prompt={class_prompt}")
        cmd.append("--with_prior_preservation")
        cmd.append(f"--prior_loss_weight={prior_loss_weight}")
        cmd.append(f"--num_class_images={num_class_images}")

    if train_text_encoder:
        cmd.append("--train_text_encoder")

    # Run the command
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    os.system(cmd_str)

    return {"status": "success", "output_dir": output_path}


@app.function(
    image=image,
    gpu="a100",
    volumes={
        "/models": model_volume,
    },
    mounts=[local_mount],
)
def sample_images(
    model_path: str,
    prompts: list,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):
    """Generate images using the fine-tuned model."""
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        KolorsPipeline,
    )

    # Check if model_path starts with a path that indicates it's in the local mount
    if model_path.startswith("dreambooth/"):
        full_model_path = os.path.join("/app", model_path)
    else:
        full_model_path = os.path.join("/models", model_path)

    # Load the pipeline

    pipe = KolorsPipeline.from_pretrained(
        full_model_path, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")

    # Enable memory optimization
    pipe.enable_xformers_memory_efficient_attention()

    # Generate images
    results = []
    for prompt in prompts:
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Convert to bytes
        import io
        from PIL import Image

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        results.append({"prompt": prompt, "image_bytes": image_bytes})

    return results


@app.local_entrypoint()
def main():
    """Run the Dreambooth training process."""
    print("Starting Dreambooth training on Modal...")

    model_name = "Kwai-Kolors/Kolors-diffusers"
    instance_dir = "kolors_training_data"  # Local directory path
    output_dir = "output/"  # Local path for output
    instance_prompt = "zsh手写的字"

    # Optional class parameters
    class_dir = None  # Set to None to disable class-based training
    class_prompt = None

    # Make sure local directories exist
    os.makedirs(instance_dir, exist_ok=True)
    if class_dir:
        os.makedirs(class_dir, exist_ok=True)
    os.makedirs(f"output", exist_ok=True)  # Create parent output directory

    # Upload model if it's a HuggingFace model ID
    # if "/" in model_name and not model_name.startswith("/"):
    #     from huggingface_hub import snapshot_download

    #     local_path = snapshot_download(model_name)

    #     # Upload to Modal volume
    #     print(f"Uploading {model_name} to Modal volume...")
    #     for root, _, files in os.walk(local_path):
    #         for file in files:
    #             source_path = os.path.join(root, file)
    #             rel_path = os.path.relpath(source_path, local_path)
    #             dest_path = os.path.join(model_name, rel_path)

    #             with open(source_path, "rb") as f:
    #                 model_volume.write(dest_path, f.read())

    # Train the model
    train_args = {
        "model_name": model_name,
        "instance_dir": instance_dir,
        "output_dir": output_dir,
        "instance_prompt": instance_prompt,
    }

    # Only add class parameters if they are provided
    if class_dir and class_prompt:
        train_args["class_dir"] = class_dir
        train_args["class_prompt"] = class_prompt
        train_args["with_prior_preservation"] = True

    result = train_dreambooth.remote(**train_args)

    print(f"Training completed! Model saved to: {result['output_dir']}")

    # Generate sample images with the trained model
    test_prompts = [
        f"a photo of {instance_prompt} in a park",
        f"a painting of {instance_prompt} in the style of Picasso",
        f"{instance_prompt} with sunglasses on the beach",
    ]

    print("Generating sample images...")
    samples = sample_images.remote(output_dir, test_prompts)

    # Save the generated images locally
    os.makedirs("dreambooth/samples", exist_ok=True)
    for i, sample in enumerate(samples):
        with open(f"dreambooth/samples/sample_{i}.png", "wb") as f:
            f.write(sample["image_bytes"])
        print(f"Saved sample with prompt: {sample['prompt']}")

    print("Done! Check the 'samples' directory for generated images.")
