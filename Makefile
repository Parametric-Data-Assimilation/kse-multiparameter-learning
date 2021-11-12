.PHONY: plots lint


PYTHON = python3
PYTEST = pytest --cov -s .
LINTER = flake8


plots:
	$(PYTHON) plots.py
lint:
	$(PYTHON) -m $(LINTER) *.py

