# Variables
PYTHON = python3
PIP = pip3
APP = main.py
VENV = venv

# Default target
all: run

# Create virtual environment
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install --upgrade pip

# Install dependencies
install: $(VENV)/bin/activate requirements.txt
	. $(VENV)/bin/activate; $(PIP) install -r requirements.txt

# Run the app
run:
	$(PYTHON) $(APP)

# Run tests
test:
	$(PYTHON) -m unittest discover -s tests

# Lint
lint:
	flake8 .

# Format
format:
	black .

# Clean up cache/pyc files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

.PHONY: all install run test lint format clean
