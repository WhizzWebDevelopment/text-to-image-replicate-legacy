#!/bin/bash
# Quick Setup Script for Text-to-Image Generator

set -e

echo "=========================================="
echo "Text-to-Image Generator - Quick Setup"
echo "=========================================="
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo ""
    echo "⚠️  Not in a virtual environment"
    echo "   Recommended: Create a venv first"
    echo ""
    read -p "Create virtual environment? (y/n): " create_venv
    
    if [[ "$create_venv" == "y" ]]; then
        echo ""
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
        echo "   ✓ Virtual environment created"
        echo ""
        echo "Activate it with:"
        echo "   source venv/bin/activate"
        echo ""
        echo "Then run this script again."
        exit 0
    fi
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo "   This may take a few minutes..."
echo ""

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed!"

# Check GPU
echo ""
echo "🖥️  Checking for GPU support..."
python3 -c "import torch; print('   ✓ GPU available:', torch.cuda.is_available())" 2>/dev/null || echo "   ⚠ Could not check GPU"

# Run test
echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test installation:"
echo "   python test_installation.py"
echo ""
echo "2. Generate an image (command line):"
echo "   python generate_local.py 'a beautiful sunset over mountains'"
echo ""
echo "3. Launch web UI:"
echo "   python gradio_ui.py"
echo ""
echo "4. Batch generate:"
echo "   python batch_generate.py example_prompts.txt"
echo ""
echo "📖 Read README.md for detailed documentation"
echo ""
