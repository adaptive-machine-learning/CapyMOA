
# Variables
NOTEBOOKS := $(wildcard notebooks/*.ipynb)

test:
	python -m pytest --durations=0 
	@echo "All Unit Tests Passed"
	@echo "Testing Notebook"
	python -m pytest --nbmake -n=auto notebooks
	@echo "All Notebook Tests Passed"

download:
	python scripts/download_datasets_and_moa.py
