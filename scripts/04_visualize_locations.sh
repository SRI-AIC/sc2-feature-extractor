#!/bin/bash

# ===============================================
# Location visualizer script for StarCraft II replays.
# author: Pedro Sequeira
# email: pedro.sequeira@sri.com
# ===============================================

# DIRECTORIES
REPLAYS_DIR="$SC2PATH/Replays/DefeatRoaches"
FEATURES_CONF="config/roaches.json"
OUTPUT_DIR="output/minigames/roaches/visualizer"

# OPTIONS
SC2_VERSION="latest"
DARK=true                    # whether to use the dark theme
IMG_FORMAT='pdf'             # file format of generated plots
FEATURE_SCREEN_SIZE="112,84" # resolution for screen feature layers
PARALLEL=-1                  # how many instances to run in parallel
VERBOSITY=1                  # logging verbosity level
CLEAR=true                   # whether to clear output directories before generating results

# ===============================================

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# run reply visualizer
echo "========================================"
echo "Visualizing unit locations for replays in '$REPLAYS_DIR'..."
python -m feature_extractor.bin.visualize_locations \
  --replays="$REPLAYS_DIR" \
  --replay_sc2_version=$SC2_VERSION \
  --config=$FEATURES_CONF \
  --output=$OUTPUT_DIR \
  --feature_screen_size=$FEATURE_SCREEN_SIZE \
  --parallel=$PARALLEL \
  --dark=$DARK \
  --format=$IMG_FORMAT \
  --verbosity=$VERBOSITY \
  --clear=$CLEAR
