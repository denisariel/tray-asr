#!/bin/bash
# Speech Recognition Tray App - macOS/Linux launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if it doesn't exist or is broken
if [ ! -f ".venv/bin/python" ]; then
    echo "🔧 Setting up virtual environment with uv..."
    uv venv
    uv pip install -r requirements.txt
    echo "✅ Setup complete!"
fi

# Run the app
.venv/bin/python main.py
