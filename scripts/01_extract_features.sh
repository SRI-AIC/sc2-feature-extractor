#!/bin/bash

# ===============================================
# High-level feature extraction script for StarCraft II replays.
# author: Pedro Sequeira
# email: pedro.sequeira@sri.com
# ===============================================

# DIRECTORIES
REPLAYS_DIR="$SC2PATH/Replays/DefeatRoaches"
FEATURES_CONF="feature-extractor/config/roaches.json"
FEATURES_DIR="output/minigames/roaches/features"

# OPTIONS
SC2_VERSION="latest"
CATEGORICAL=false # whether to extract categorical (vs numerical) features
KEEP_CSV=false    # whether to keep individual CSV files after creating zips
AMOUNT=100        # number of replays for which to extract features
PARALLEL=-1       # how many instances to run in parallel
VERBOSITY=1       # logging verbosity level
CLEAR=true        # whether to clear output directories before generating results

# ===============================================

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../.." || exit
clear

echo "========================================"
echo "Extracting features from replays in '$REPLAYS_DIR'..."
python -m feature_extractor.bin.extract_features \
  --replay_sc2_version=$SC2_VERSION \
  --categorical=$CATEGORICAL \
  --config=$FEATURES_CONF \
  --replays=$REPLAYS_DIR \
  --amount=$AMOUNT \
  --output=$FEATURES_DIR \
  --parallel=$PARALLEL \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY \
  --keep_csv=$KEEP_CSV
