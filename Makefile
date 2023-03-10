install: ## [Local development, CPU] Upgrade pip, install requirements, install package.
	python -m pip install -U pip && python -m pip install -U setuptools wheel
	python -m pip install -e "."

install-dev: ## [Local development] install test requirements
	python -m pip install -r test-requirements.txt

lint:
	python -m pylint -j 0 restless/

black:
	python -m black -l 110 restless/

mypy:
	python -m mypy --ignore-missing-imports restless/

test:
	python -m pytest tests/

coverage:
	coverage run -m pytest tests/
	coverage report -m

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
