#!/bin/bash
# SETUP SCRIPT FOR ITviec Reviews Project

echo "🔹 Creating virtual environment..."
python3 -m venv .venv

echo "🔹 Activating virtual environment..."
source .venv/bin/activate

echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing requirements..."
pip install -r requirements.txt

echo "✅ Setup complete! Activate your environment using: source .venv/bin/activate"
