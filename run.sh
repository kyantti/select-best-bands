#!/bin/bash

# Ensure the script is executable
# Run `chmod +x run.sh` before executing

# Define the range of experiments
START_EXPERIMENT=11
END_EXPERIMENT=20

# Loop through the range and execute the Python script for each experiment sequentially
for EXPERIMENT_NUM in $(seq $START_EXPERIMENT $END_EXPERIMENT); do
    echo "Running experiment $EXPERIMENT_NUM..."
    uv run python -u ga.py $EXPERIMENT_NUM > out/logs/experiment_${EXPERIMENT_NUM}.log 2>&1
    echo "Experiment $EXPERIMENT_NUM completed. Logs: out/logs/experiment_${EXPERIMENT_NUM}.log"
done

echo "All experiments completed."
echo "All experiments completed."
