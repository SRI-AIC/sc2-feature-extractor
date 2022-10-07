#!/bin/bash

# ===============================================
# Video recording script for StarCraft II replays.
# author: Pedro Sequeira
# email: pedro.sequeira@sri.com
# ===============================================

# DIRECTORIES
REPLAYS_DIR="$SC2PATH/Replays/DefeatRoaches"
OUTPUT_DIR="output/minigames/roaches/videos"

# OPTIONS
SC2_VERSION="latest"
FPS=10                # the fps ratio used to save the videos.
CRF=18                # video constant rate factor: the default quality setting in [0, 51]
PARALLEL=1            # do not change this, only 1 replay at a time!
WINDOW_SIZE="800,600" # resolution/size of SC2 game window
HIDE_HUD=True         # whether to hide the HUD / information panel at the bottom of the screen.
VERBOSITY=1           # logging verbosity level
CLEAR=true            # whether to clear output directories before generating results

# ===============================================

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# run reply visualizer
echo "========================================"
echo "Recording videos for replays in '$REPLAYS_DIR'..."
python -m feature_extractor.bin.record_videos \
  --replays="$REPLAYS_DIR" \
  --replay_sc2_version=$SC2_VERSION \
  --output=$OUTPUT_DIR \
  --window_size=$WINDOW_SIZE \
  --fps $FPS \
  --crf $CRF \
  --parallel=$PARALLEL \
  --hide_hud=$HIDE_HUD \
  --verbosity=$VERBOSITY \
  --clear=$CLEAR
