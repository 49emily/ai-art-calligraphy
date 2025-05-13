# Training CycleGAN and pix2pix on Modal

This document explains how to use [Modal](https://modal.com/) for training CycleGAN and pix2pix models. Modal provides easy access to cloud GPUs and simplifies the deployment process.

## Prerequisites

1. Install Modal: `pip install modal`
2. Sign up for Modal and authenticate: `modal token new`
3. (Optional) Configure your WANDB API key if you want to use Weights & Biases:
   ```
   modal secret create wandb WANDB_API_KEY=your_wandb_api_key
   ```

## Training Methods

There are two ways to train your models on Modal:

1. **Direct Upload**: Code and data are uploaded when you run the training script
2. **Volume Mounting**: Datasets are pre-uploaded to Modal volumes for faster startup and reuse

## Method 1: Direct Upload

This method uploads your code and dataset when you run the training script. It's simpler but slower for large datasets.

```bash
# From the tools/pytorch-CycleGAN-and-pix2pix directory:
modal run modal_train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```

## Method 2: Using Modal Volumes (Recommended)

This method pre-uploads your datasets to Modal volumes, which is faster for large datasets and allows reuse.

### Step 1: Upload your dataset to a Modal volume

```bash
# Upload a dataset to a Modal volume
python modal_data_uploader.py --dataset_name facades --dataset_path ./datasets/facades
```

### Step 2: Train using the volume

```bash
# Train using the volume (much faster startup)
modal run modal_train.py --volume_name cyclegan-facades --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```

## GPU Selection

Modal offers different GPU types. You can select the GPU type with the `--gpu_type` parameter:

```bash
# Train with an A100 GPU
modal run modal_train.py --volume_name cyclegan-facades --gpu_type A100 --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```

Available GPU types:

- `T4` (default): Good balance of cost and performance
- `A10G`: Better performance for larger models
- `A100`: High performance for large models and datasets
- `H100`: Highest performance (most expensive)

## Training Options

The Modal training script supports all the options from the original CycleGAN and pix2pix framework.

Here are some common examples:

### Training CycleGAN:

```bash
modal run modal_train.py --volume_name cyclegan-maps --dataroot /datasets/maps --name maps_cyclegan --model cycle_gan
```

### Training pix2pix:

```bash
modal run modal_train.py --volume_name cyclegan-facades --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```

### Continuing Training:

```bash
modal run modal_train.py --volume_name cyclegan-facades --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --continue_train --epoch_count 20
```

### Using Weights & Biases:

```bash
modal run modal_train.py --volume_name cyclegan-facades --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --use_wandb
```

## Customizing Batch Size and Epochs

You can adjust the batch size and number of epochs to optimize training:

```bash
modal run modal_train.py --volume_name cyclegan-facades --dataroot /datasets/facades --name facades_pix2pix --model pix2pix --batch_size 4 --n_epochs 100 --n_epochs_decay 100
```

## Retrieving Results

After training, your model checkpoints will be saved in the Modal environment. You can download them using:

```bash
modal volume get cyclegan-output /path/to/local/directory
```

Or set up a shared volume to automatically save results:

```bash
# Create a volume for outputs
modal volume create cyclegan-output

# Use it when training
modal run modal_train.py --volume_name cyclegan-facades --output_volume cyclegan-output --dataroot /datasets/facades --name facades_pix2pix --model pix2pix
```

## Troubleshooting

1. **Connection Issues**: If you encounter network problems, try re-authenticating with `modal token new`
2. **Memory Errors**: Try reducing batch size or using a larger GPU type
3. **Dataset Errors**: Ensure your dataset follows the expected directory structure
4. **Missing Dependencies**: The image should have all required dependencies, but if you need additional ones, modify the `image` definition in `modal_train.py`
