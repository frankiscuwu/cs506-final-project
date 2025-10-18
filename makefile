# Use Python 3.12 by default for tensorflow
PYTHON ?= python3.12
VENV = venv

# Default target
all: install

# Check Python version (must be <= 3.12)
check-python:
	@echo "Checking Python version..."
	@$(PYTHON) -c "import sys; v=sys.version_info[:2]; assert v <= (3,12), f'Python {v[0]}.{v[1]} is not supported. Use <=3.12.'"

# Create virtual environment
$(VENV)/bin/python: check-python
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip

# Install dependencies and download dataset
install: $(VENV)/bin/python requirements.txt
	@echo "Installing Python packages..."
	$(VENV)/bin/python -m pip install -r requirements.txt
	@echo "Downloading dataset..."
	KAGGLEHUB_CACHE=data/raw/ALL $(VENV)/bin/python src/download_dataset.py

# Run tests
test:
	$(VENV)/bin/python -m unittest discover -s tests

# Lint code
lint:
	$(VENV)/bin/python -m flake8 .

# Format code
format:
	$(VENV)/bin/python -m black .

# Clean cache/pyc files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

.PHONY: all check-python install test lint format clean
