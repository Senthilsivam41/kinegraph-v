#!/bin/bash

# Demo script to run validation tests for the three improvements
echo "=========================================="
echo "Running Multi-Hop Traversal Improvement Validation"
echo "=========================================="
echo ""
echo "This demonstrates all three improvements with sample data."
echo ""

cd "$(dirname "$0")/.." || exit 1

# Run the validation script
python scripts/validate_improvements.py --demo
