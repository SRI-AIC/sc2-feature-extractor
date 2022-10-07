#!/bin/bash

# ===============================================
# Feature descriptor script for StarCraft II high-level feature extractor.
# author: Pedro Sequeira
# email: pedro.sequeira@sri.com
# ===============================================

# DIRECTORIES
FEATURES_CONF="config/roaches.json"
OUTPUT_DIR="output/minigames/roaches/descriptor"

# OPTIONS
VERBOSITY=1       # logging verbosity level
CLEAR=true        # whether to clear output directories before generating results

# ===============================================

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# run feature descriptor
python -m feature_extractor.bin.feature_descriptor \
  --config=$FEATURES_CONF \
  --output=$OUTPUT_DIR \
  --verbosity=$VERBOSITY \
  --clear=$CLEAR
