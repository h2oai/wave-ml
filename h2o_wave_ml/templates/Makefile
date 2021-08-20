SHELL      := bash
PYTHON     := python3

.DEFAULT_GOAL := help

venv:
	$(PYTHON) -m venv venv

.PHONY: setup
setup: | venv ## Setup a development environment
	./venv/bin/python -m pip install -U pip
	./venv/bin/python -m pip install h2o-wave[ml]

.PHONY: run
run: ## Run application
	./venv/bin/wave run index.py

.PHONY: clean
clean: ## Clean all files produced by make
	rm -rf venv

.PHONY: help
help: ## List all make tasks
	@grep -Eh '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
