SHELL      := bash
PYTHON     := python3

.DEFAULT_GOAL := help

.PHONY: setup
setup: venv ## Setup a development environment

venv:
	$(PYTHON) -m venv venv
	./venv/bin/python -m pip install -U pip
	./venv/bin/python -m pip install wheel
	./venv/bin/python -m pip install --editable .

.PHONY: clean
clean: ## Clean files produced by make
	rm -rf venv

.PHONY: release
release: venv ## Create a .whl file
	./venv/bin/python setup.py bdist_wheel

.PHONY: help
help: ## List all make tasks
	@grep -Eh '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
