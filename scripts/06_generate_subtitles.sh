#!/bin/bash

# ===============================================
# Subtitle generation script for StarCraft II replays.
# author: Pedro Sequeira
# email: pedro.sequeira@sri.com
# ===============================================

# DIRECTORIES
INPUT_FILE="output/minigames/roaches/features/feature-dataset.pkl.gz"
OUTPUT_DIR="output/minigames/roaches/videos"

# OPTIONS
FEATURES="Advancing_Friendly_Marine_Roach Attacking_Friendly_Blue_Roach Distance_Marine_Roach Number_Friendly_Marine Number_Enemy_Roach"
PARALLEL=-1 # num parallel processes
VERBOSITY=1 # logging verbosity level
CLEAR=false # whether to clear output directories before generating results

# ===============================================

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# run reply visualizer
echo "========================================"
echo "Generating subtitles for features in '$INPUT_FILE'..."
python -m feature_extractor.bin.generate_subtitles \
  --input=$INPUT_FILE \
  --output=$OUTPUT_DIR \
  --features $FEATURES \
  --parallel=$PARALLEL \
  --verbosity=$VERBOSITY \
  --clear=$CLEAR
