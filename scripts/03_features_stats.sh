#!/bin/bash

# ===============================================
# Feature stats script for StarCraft II high-level features.
# author: Pedro Sequeira
# email: pedro.sequeira@sri.com
# ===============================================

# DIRECTORIES
FEATURES_FILE="output/minigames/roaches/features/feature-dataset.pkl.gz"
FEATURES_DESC_FILE="output/minigames/roaches/descriptor/roaches_desc.json"
OUTPUT_DIR="output/minigames/roaches/stats"

# OPTIONS
IMG_FORMAT='pdf' # file format of generated plots
PARALLEL=-1      # how many instances to run in parallel
VERBOSITY=1      # logging verbosity level
CLEAR=true       # whether to clear output directories before generating results

# ===============================================

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../.." || exit
clear

# run feature descriptor
echo "========================================"
echo "Analyzing features stats for '$FEATURES_FILE'..."
python -m feature_extractor.bin.feature_stats \
  --input=$FEATURES_FILE \
  --desc=$FEATURES_DESC_FILE \
  --output=$OUTPUT_DIR \
  --format=$IMG_FORMAT \
  --parallel=$PARALLEL \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY
