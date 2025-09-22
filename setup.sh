#!/bin/bash

# BSS Maitri Installation and Setup Script

echo "🚀 Setting up BSS Maitri AI Assistant"
echo "====================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -o "3\.[0-9]*")
if [[ $(echo "$python_version" | cut -d. -f2) -lt 8 ]]; then
    echo "❌ Python 3.8 or higher is required. Current version: $(python3 --version)"
    exit 1
fi
echo "✅ Python version: $(python3 --version)"

# Check if Ollama is installed
echo "📋 Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install Ollama first:"
    echo "   Visit: https://ollama.ai/download"
    echo "   Or run: curl -fsSL https://ollama.ai/install.sh | sh"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Ollama is installed"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "📦 Installing BSS Maitri..."
pip install -e .

# Install development dependencies (optional)
read -p "Install development dependencies? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[dev]"
fi

# Setup Ollama model
echo "🤖 Setting up Ollama model..."
python -m bss_maitri.main --mode setup

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Quick Start:"
echo "   source venv/bin/activate"
echo "   bss-maitri --mode web"
echo ""
echo "🌐 This will start the web interface at http://localhost:7860"
echo ""
echo "📚 Other options:"
echo "   bss-maitri --mode cli     # Command line interface"
echo "   bss-maitri --help        # Show all options"