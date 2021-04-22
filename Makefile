SHELL      := bash
PYTHON     := python3

.DEFAULT_GOAL := help

venv:
	$(PYTHON) -m venv venv

.PHONY: setup
setup: | venv ## Setup a development environment
	./venv/bin/python -m pip install -U pip
	./venv/bin/python -m pip install setuptools wheel
	./venv/bin/python -m pip install --editable .

.PHONY: purge
purge: ## Purge previous build but keep `venv`
	rm -rf build dist h2o_wave_ml.egg-info

.PHONY: clean
clean: purge ## Clean all files produced by make
	rm -rf venv

.PHONY: release
release: setup ## Create a .whl file
	./venv/bin/python setup.py bdist_wheel

.PHONY: release-pypi
release-pypi: setup
	sed -e "/h2osteam/d" -e "/mlops-client/d" setup.py > setup.alt.py
	./venv/bin/python setup.alt.py bdist_wheel

.PHONY: help
help: ## List all make tasks
	@grep -Eh '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
