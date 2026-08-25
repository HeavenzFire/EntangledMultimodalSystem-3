#!/bin/bash
# Development Setup Script - Optimized for Quick Onboarding
# Usage: ./setup-dev.sh

set -e

echo "🚀 Starting Development Environment Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    echo "Upgrading pip..."
    pip install --upgrade pip wheel setuptools
    print_success "Pip upgraded"
}

# Install dependencies
install_deps() {
    echo "Installing optimized dependencies..."
    if [ -f "requirements-optimized.txt" ]; then
        pip install -r requirements-optimized.txt
        print_success "Optimized dependencies installed"
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "No requirements file found"
        exit 1
    fi
}

# Install pre-commit hooks
setup_precommit() {
    if [ -f ".pre-commit-config.yaml" ]; then
        echo "Setting up pre-commit hooks..."
        pip install pre-commit
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "No pre-commit config found, skipping..."
    fi
}

# Initialize submodules
init_submodules() {
    echo "Initializing git submodules..."
    git submodule update --init --recursive
    print_success "Submodules initialized"
}

# Run tests
run_tests() {
    echo "Running quick test suite..."
    pytest tests/ -x -v --tb=short || print_warning "Some tests failed"
}

# Show next steps
show_next_steps() {
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run the test suite:"
    echo "   pytest tests/ -n auto"
    echo ""
    echo "3. Start development:"
    echo "   python src/main.py"
    echo ""
    echo "4. Run linting:"
    echo "   pre-commit run --all-files"
    echo ""
    echo "5. Build documentation:"
    echo "   sphinx-build docs/ docs/_build/"
    echo ""
    echo "For more optimization tips, see DEVELOPMENT_OPTIMIZATION.md"
    echo "=========================================="
}

# Main setup flow
main() {
    check_python
    setup_venv
    upgrade_pip
    install_deps
    setup_precommit
    init_submodules
    
    echo ""
    echo "Setup complete! Running quick validation..."
    run_tests
    
    show_next_steps
}

# Run main function
main "$@"
