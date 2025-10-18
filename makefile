# Variables
PYTHON = python3
PIP = pip3
# APP = main.py
VENV = venv

# Default target
all: install

# Create virtual environment
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install --upgrade pip

# Run installation and download
install: $(VENV)/bin/activate requirements.txt
	. $(VENV)/bin/activate; $(PIP) install -r requirements.txt
	# Set Kagglehub cache and run the download script
	. $(VENV)/bin/activate; KAGGLEHUB_CACHE=data/raw/ALL python src/download_dataset.py

# Run the app
# run:
# 	$(PYTHON) $(APP)

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
