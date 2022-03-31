#!/bin/bash

# ===============================================
# StarCraft II replay generation scrip for the DefeatRoaches task.
# author: Pedro Sequeira
# email: pedro.sequeira@sri.com
# ===============================================

# OPTIONS
AMOUNT=100
TASK="DefeatRoaches"
AGENT=pysc2.agents.scripted_agent.DefeatRoaches

# ===============================================

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../.." || exit
clear

echo "========================================"
echo "Generating $AMOUNT replays for the '$TASK' task..."
for ((i = 1; i <= $AMOUNT; i++)); do
  python -m pysc2.bin.agent --map $TASK --agent $AGENT --render=False --max_episodes 1
done
