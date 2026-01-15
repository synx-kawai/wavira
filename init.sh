#!/bin/bash
# Wavira Project Initialization Script
# Run this at the start of each coding session to verify environment state

set -e

echo "=== Wavira Project Initialization ==="
echo ""

# Check Python environment
echo "1. Checking Python environment..."
if command -v python3 &> /dev/null; then
    echo "   Python: $(python3 --version)"
else
    echo "   ERROR: Python3 not found"
    exit 1
fi

# Check if in virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "   Virtual env: $VIRTUAL_ENV"
else
    echo "   WARNING: Not in virtual environment"
    if [ -d "venv" ]; then
        echo "   Activating venv..."
        source venv/bin/activate
    fi
fi

# Install dependencies
echo ""
echo "2. Installing dependencies..."
pip install -e . -q 2>/dev/null || pip install -e ".[dev]" -q 2>/dev/null || echo "   Dependencies already installed"

# Run basic tests
echo ""
echo "3. Running basic tests..."
if python -m pytest tests/ -v --tb=short -q 2>/dev/null; then
    echo "   ✓ All tests passed"
else
    echo "   ✗ Some tests failed - check output above"
fi

# Git status
echo ""
echo "4. Git status..."
echo "   Branch: $(git branch --show-current)"
echo "   Status:"
git status --short

# Show feature status
echo ""
echo "5. Feature status (from features.json)..."
if [ -f "features.json" ]; then
    python3 -c "
import json
with open('features.json') as f:
    data = json.load(f)
for feat in data['features']:
    status_icon = '✓' if feat['status'] == 'passing' else '✗'
    print(f\"   {status_icon} [{feat['priority']}] #{feat['issue']}: {feat['title']} - {feat['status']}\")
"
else
    echo "   features.json not found"
fi

# Show recent progress
echo ""
echo "6. Recent progress (last 5 commits)..."
git log --oneline -5

echo ""
echo "=== Initialization Complete ==="
echo "Ready to work on features. Check claude-progress.txt for context."
