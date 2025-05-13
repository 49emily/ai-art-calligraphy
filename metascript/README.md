# Image Upscaling Tool

This script processes a dataset of images using the FAL.ai ESRGAN model for upscaling. It allows you to batch process multiple images at once, with rate limiting and progress tracking.

## Requirements

- Python 3.6+
- Pip packages: `requests`, `pillow`

## Setup

1. Install required packages:

   ```
   pip install requests pillow
   ```

2. Set up FAL.ai API credentials as environment variables:

   ```
   export FAL_API_KEY=your_api_key
   export FAL_API_SECRET=your_api_secret
   ```

   You can obtain these from the [FAL.ai website](https://fal.ai/).

## Usage

```
python process_dataset.py --dataset <path_to_images> --output <output_directory>
```

### Options

- `--dataset`: Directory containing images to upscale (required)
- `--output`: Directory to save upscaled images (required)
- `--force`: Process images even if output already exists (default: skip existing)
- `--rate-limit`: Maximum requests per minute (default: 5)

### Example

```
python process_dataset.py --dataset ./my_photos --output ./upscaled_photos --rate-limit 10
```

## Features

- Automatic image format detection
- Large image handling with resize option
- Progress tracking with stats
- Rate limiting to prevent API throttling
- Skip already processed images
- Detailed error reporting

## Notes

- Large images are automatically resized to reduce upload size
- Upscaled images are saved with the prefix "upscaled\_" in the output directory
- The script will create the output directory if it doesn't exist
- The default rate limit is 5 requests per minute (adjust based on your API plan)
