# Variables
PYTHON ?= python3
VENV = venv

# Default target
all: install

# Create virtual environment
$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip

# Install dependencies and download dataset
install: $(VENV)/bin/python requirements.txt
	$(VENV)/bin/python -m pip install -r requirements.txt
	KAGGLEHUB_CACHE=data/raw $(VENV)/bin/python src/download_dataset.py

# Run tests
test:
	$(VENV)/bin/python -m unittest discover -s tests

# Lint
lint:
	$(VENV)/bin/python -m flake8 .

# Format
format:
	$(VENV)/bin/python -m black .

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

.PHONY: all install test lint format clean
