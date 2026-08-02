# Quick Start Guide - Development Acceleration

## 🚀 Get Started in 5 Minutes

### Option 1: Automated Setup (Recommended)

```bash
# Clone and setup
git clone <your-repo-url>
cd <your-repo>
./setup-dev.sh
```

This will:
- Create a virtual environment
- Install optimized dependencies
- Set up pre-commit hooks
- Initialize submodules
- Run validation tests

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install optimized dependencies
pip install -r requirements-optimized.txt

# Install pre-commit
pip install pre-commit
pre-commit install

# Initialize submodules
git submodule update --init --recursive
```

## ⚡ Development Commands

### Testing
```bash
# Run all tests (parallelized)
pytest tests/ -n auto

# Run specific test file
pytest tests/test_specific.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run benchmarks
pytest tests/ --benchmark-only
```

### Code Quality
```bash
# Format code
black src/ core/ tests/

# Sort imports
isort src/ core/ tests/

# Type checking
mypy src/ core/

# Linting
flake8 src/ core/ tests/

# Run all pre-commit checks
pre-commit run --all-files
```

### Building
```bash
# Build package
python -m build

# Check package
twine check dist/*

# Install in development mode
pip install -e .
```

### Documentation
```bash
# Build docs
sphinx-build docs/ docs/_build/

# Serve docs locally
cd docs/_build && python -m http.server 8000
```

## 🎯 Performance Tips

### 1. Parallel Testing
Use `pytest-xdist` to run tests in parallel:
```bash
pytest tests/ -n auto  # Auto-detect CPU count
pytest tests/ -n 4     # Use 4 workers
```

### 2. Incremental Builds
Only test changed files:
```bash
# Using pytest with path filtering
pytest tests/ -k "test_feature_x"

# Using git to find changed test files
git diff --name-only main | grep test_ | xargs pytest
```

### 3. Caching
Enable pip caching for faster installs:
```bash
export PIP_CACHE_DIR=~/.cache/pip
```

### 4. Pre-commit for Fast Feedback
Pre-commit catches issues before CI:
```bash
# Install
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🔧 Optimization Checklist

### Immediate Actions (Day 1)
- [ ] Run `./setup-dev.sh`
- [ ] Review DEVELOPMENT_OPTIMIZATION.md
- [ ] Set up IDE with recommended extensions
- [ ] Configure pre-commit hooks

### Week 1
- [ ] Familiarize with codebase structure
- [ ] Run full test suite
- [ ] Review CODEOWNERS for your area
- [ ] Set up local debugging

### Month 1
- [ ] Contribute to first PR
- [ ] Understand CI/CD pipeline
- [ ] Review performance benchmarks
- [ ] Suggest optimizations

## 📊 Metrics to Watch

| Metric | Target | Current |
|--------|--------|---------|
| CI Build Time | < 5 min | See workflows |
| Test Suite Time | < 10 min | Run locally |
| Code Coverage | > 80% | Check reports |
| First Commit Time | < 30 min | Track onboarding |

## 🛠️ Troubleshooting

### Common Issues

**Import errors after setup:**
```bash
# Reinstall dependencies
pip install -r requirements-optimized.txt --force-reinstall
```

**Pre-commit hooks failing:**
```bash
# Update pre-commit
pre-commit autoupdate
pre-commit run --all-files
```

**Tests failing randomly:**
```bash
# Run without parallelization to debug
pytest tests/ -n 0 -v
```

**Submodule issues:**
```bash
# Reinitialize submodules
git submodule sync
git submodule update --init --recursive --force
```

## 📚 Additional Resources

- **Full Optimization Plan**: See `DEVELOPMENT_OPTIMIZATION.md`
- **Contributing Guidelines**: See `CONTRIBUTING.md`
- **Code of Conduct**: See `CODE_OF_CONDUCT.md`
- **Security Policy**: See `SECURITY.md`

## 🤝 Getting Help

- Open an issue for bugs
- Discussion tab for questions
- Check existing documentation first
- Tag relevant team members using CODEOWNERS

---

**Ready to accelerate development?** Start with `./setup-dev.sh` and you'll be coding in minutes!
