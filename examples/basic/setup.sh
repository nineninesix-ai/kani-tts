#!/bin/bash
set -e

echo "=== KaniTTS Setup ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Validate Python version (only 3.10-3.12 supported)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 12 ]; then
    echo "Error: This project requires Python 3.10, 3.11, or 3.12"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi
echo "Python version is supported"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created successfully"
else
    echo ""
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Add dependencies
echo ""
echo "Adding dependencies..."

# Install FastAPI and Uvicorn
echo "Installing FastAPI and Uvicorn..."
pip install fastapi uvicorn

# Install nemo-toolkit (which will install transformers 4.53)
echo ""
echo "Installing nemo-toolkit[tts]..."
pip install "nemo-toolkit[tts]==2.4.0"

# Force reinstall transformers to 4.57.1 (required for model compatibility)
echo ""
echo "Upgrading transformers to 4.57.1..."
echo "Note: nemo-toolkit[tts] requires transformers==4.53, but we need 4.57.1 for model compatibility"
pip install "transformers==4.57.1"

# Verify installation
echo ""
echo "=== Verifying Installation ==="
echo ""

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('CUDA not available')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now start the server with:"
echo "  source venv/bin/activate"
echo "  python server.py"
echo ""
echo "Note: Models will be automatically downloaded on first run (~1.5GB)"
