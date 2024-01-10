
# Variables
NOTEBOOKS := $(wildcard notebooks/*.ipynb)

run_all_notebooks:
	@echo "Running all notebooks..."
	@for notebook in $(NOTEBOOKS); do \
		jupyter nbconvert --inplace --to notebook --execute $$notebook; \
	done
	@echo "Done."

download:
	python scripts/download_datasets_and_moa.py
