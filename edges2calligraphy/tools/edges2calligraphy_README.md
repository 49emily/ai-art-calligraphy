# Strokes to Calligraphy with Pix2Pix

This tool converts Chinese characters to calligraphy-style handwriting using a trained pix2pix model. The process works by:

1. Rendering a character using Source Han Sans SC font (ultra-thin weight)
2. Extracting the actual strokes of the rendered character
3. Applying a pix2pix model to transform these strokes into handwriting

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- PIL/Pillow
- NumPy
- PyTorch (for the pix2pix model)
- Source Han Sans SC Variable Font installed on your system (or another font of your choice)

## Usage

### Using the Shell Script

The easiest way to use this tool is with the provided shell script:

```bash
# Single character
./tools/generate_handwriting.sh -c "字"

# Multiple characters
./tools/generate_handwriting.sh -c "你好"
```

Options:

- `-c CHARACTERS`: Single character or string of characters to render (required)
- `-s FONT_SIZE`: Font size for rendering (default: 128)
- `-m MODEL_NAME`: Name of the trained pix2pix model (default: characters)
- `-f FONT_PATH`: Path to the font file (default: /Library/Fonts/SourceHanSansSC-VF.otf)
- `-w WEIGHT`: Font weight (default: 100, 100=ultra thin, 900=heavy) for variable fonts
- `-t THICKNESS`: Stroke thickness adjustment (default: 1, higher = thinner strokes)
- `-e`: Extract strokes only (don't apply pix2pix model)

### Using the Python Script Directly

You can also use the Python script directly for more control:

```bash
# Single character
python character_to_handwriting.py --character "字" --font_size 128

# Multiple characters
python character_to_handwriting.py --character "你好世界" --font_weight 100
```

Additional options:

- `--model_name`: Name of the trained pix2pix model (default: edges2calligraphy_pix2pix)
- `--checkpoints_dir`: Directory where model checkpoints are stored (default: pytorch-CycleGAN-and-pix2pix/checkpoints)
- `--results_dir`: Directory to save the results (default: ./results)
- `--temp_dir`: Directory for temporary files (default: ./temp)
- `--font_path`: Path to the font file (default: /Library/Fonts/SourceHanSansSC-VF.otf)
- `--font_weight`: Font weight for variable fonts (default: 100, 100=ultra thin, 900=heavy)
- `--stroke_thickness`: Stroke thickness adjustment (default: 1, higher = thinner strokes)
- `--only_extract`: Only extract strokes without applying pix2pix model

## Results

The generated handwriting images will be saved in character-specific folders within the results directory (default: `./results`):

```
results/
├── char_你_0/              # Results for first character
│   ├── 你_strokes.png      # Character stroke image
│   ├── index.html          # HTML viewer for results
│   └── images/             # Generated images from pix2pix
│       ├── edges.png       # Input edge image
│       ├── fake_B.png      # Generated handwriting
│       └── real_A.png      # Original image
├── char_好_1/              # Results for second character
│   └── ...
└── ...
```

Each character gets its own folder with:

- The stroke image for the character
- A results index.html (if pix2pix was applied)
- Generated handwriting images (if pix2pix was applied)

## Examples

```bash
# Generate handwriting for a single character
./tools/generate_handwriting.sh -c "爱"

# Process multiple characters
./tools/generate_handwriting.sh -c "你好世界"

# Extract strokes only (without applying pix2pix) for debugging
./tools/generate_handwriting.sh -c "爱" -e

# Adjust font weight and stroke thickness
./tools/generate_handwriting.sh -c "爱" -w 200 -t 2
```
