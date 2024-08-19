PYTHON_ENV = env
PYTHON = $(PYTHON_ENV)/bin/python
PIP = $(PYTHON_ENV)/bin/pip
PYLINT = $(PYTHON_ENV)/bin/pylint
BLACK = $(PYTHON_ENV)/bin/black

# Define your source files
SOURCES = $(shell find . -path ./env -prune -o -name '*.py' -print)

# Create a Python virtual environment
.PHONY: env
env:
	if [ ! -d "$(PYTHON_ENV)" ]; then python3 -m venv $(PYTHON_ENV); fi
	$(PIP) install --upgrade pip

# Install dependencies
.PHONY: install
install: 
	$(PIP) install -r requirements.txt

# Run pylint on all Python files
.PHONY: lint
lint:
	@echo "Running pylint on all Python files..."
	$(PYLINT) $(SOURCES)

# Run black for code formatting on all Python files
.PHONY: format
format: 
	@echo "Running black on all Python files..."
	$(BLACK) $(SOURCES)

# Clean up environment
.PHONY: clean
clean:
	rm -rf $(PYTHON_ENV)
