#!/bin/bash

set -e

# Virtual environment location
VENV_DIR="$HOME/mindrove_venv"

echo "========================================"
echo "   MindRove SDK Installation Script"
echo "========================================"

# Check that Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

echo "Python version:"
python3 --version

# Check whether venv is available
if ! python3 -m venv --help &> /dev/null; then
    echo "Error: Python venv is not available."
    echo "On Ubuntu, install it with:"
    echo "sudo apt install python3-venv"
    exit 1
fi

# Create or update the virtual environment so it can use ROS's system packages.
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
else
    echo ""
    echo "Updating virtual environment..."
fi

python3 -m venv --system-site-packages --upgrade "$VENV_DIR"

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install MindRove SDK
echo ""
echo "Installing MindRove SDK..."
python -m pip install mindrove


# Verify installation
echo ""
echo "Verifying installation..."

if python -m pip show mindrove > /dev/null 2>&1; then
    echo "MindRove SDK installed successfully!"
    python -m pip show mindrove
else
    echo "Error: MindRove SDK installation could not be verified."
    exit 1
fi

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "To activate the environment in a new terminal, run:"
echo "source ~/mindrove_venv/bin/activate"
