#!/bin/bash
set -e

# Usage information
usage() {
  echo "Usage: $0 -c CHARACTERS [-s FONT_SIZE] [-m MODEL_NAME] [-f FONT_PATH] [-t THICKNESS] [-w WEIGHT] [-e]"
  echo
  echo "Options:"
  echo "  -c CHARACTERS   Single character or string of characters to render (required)"
  echo "  -s FONT_SIZE    Font size (default: 128)"
  echo "  -m MODEL_NAME   Name of trained pix2pix model (default: characters)"
  echo "  -f FONT_PATH    Path to font file (default: /Library/Fonts/SourceHanSansSC-VF.otf)"
  echo "  -w WEIGHT       Font weight for variable fonts (default: 100, 100=ultra thin, 900=heavy)"
  echo "  -t THICKNESS    Stroke thickness adjustment (default: 1, higher = thinner)"
  echo "  -e              Extract strokes only (don't apply pix2pix model)"
  echo
  echo "Examples:"
  echo "  $0 -c \"字\"           # Process a single character"
  echo "  $0 -c \"你好世界\"      # Process multiple characters"
  echo "  $0 -c \"字\" -e        # Extract strokes only without applying pix2pix"
  echo
  exit 1
}

# Default values
CHARACTER="涵"
FONT_SIZE=128
MODEL_NAME="characters"
FONT_PATH="/Library/Fonts/SourceHanSansSC-VF.otf"
THICKNESS=1
FONT_WEIGHT=100
EXTRACT_ONLY=""

# Parse command line arguments
while getopts "c:s:m:f:t:w:eh" opt; do
  case $opt in
    c) CHARACTER="$OPTARG" ;;
    s) FONT_SIZE="$OPTARG" ;;
    m) MODEL_NAME="$OPTARG" ;;
    f) FONT_PATH="$OPTARG" ;;
    w) FONT_WEIGHT="$OPTARG" ;;
    t) THICKNESS="$OPTARG" ;;
    e) EXTRACT_ONLY="--only_extract" ;;
    h) usage ;;
    *) usage ;;
  esac
done

# Check if character is provided
if [ -z "$CHARACTER" ]; then
  echo "Error: Character(s) are required."
  usage
fi

# Run the Python script with the provided arguments
python character_to_handwriting.py \
  --character "$CHARACTER" \
  --font_size "$FONT_SIZE" \
  --model_name "$MODEL_NAME" \
  --font_path "$FONT_PATH" \
  --font_weight "$FONT_WEIGHT" \
  --stroke_thickness "$THICKNESS" \
  $EXTRACT_ONLY

echo "Done! Check the results directory for the generated output." 