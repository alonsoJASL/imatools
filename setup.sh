#!/bin/bash

# Define environment and Python path setup file
SETUP_FILE="IMATOOLS_ENV_SETUP"
CURRENT_DIR=$(dirname "$(realpath "$0")")

echo "Creating environment setup script..."

# Write to the setup file
cat <<EOL > $SETUP_FILE
# Activate the conda environment
conda activate imatools

# Get the absolute path of the script's directory
SCRIPT_DIR=$CURRENT_DIR

# Add the script directory to PYTHONPATH
export PYTHONPATH=\$PYTHONPATH:\$SCRIPT_DIR

echo "Environment activated and PYTHONPATH set to \$SCRIPT_DIR."
EOL

# Make it clear to the user
echo "Setup complete. Run 'source $SETUP_FILE' to activate the environment and set the PYTHONPATH."
