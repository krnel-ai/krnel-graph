.PHONY: test test-fast test-minimal-import test-lowest-deps test-lowest-deps-fast test-ci docs docs-serve docs-clean docs-autobuild

test:
	@uv run --locked --all-extras pytest -v
test-fast:
	@uv run --locked --all-extras pytest -v -m "not ml_models"

test-minimal-import:
	@uv run --isolated --exact --python=3.10 --resolution=lowest-direct python -c 'import krnel; print("Minimal import success")'
test-lowest-deps:
	@uv run --isolated --exact --all-extras --python=3.10 --resolution=lowest-direct pytest -v
test-lowest-deps-fast:
	@uv run --isolated --exact --all-extras --python=3.10 --resolution=lowest-direct pytest -v -m "not ml_models"

test-ci: test-minimal-import test-lowest-deps-fast test-fast
	@echo CI tests OK

test-cov:
	@uv run --locked --all-extras pytest -v \
		--cov=src/krnel \
		--cov-report=term \
		--cov-report=xml

docs:
	@uv run --locked --extra docs sphinx-build -b html docs docs/_build/html

docs-autobuild:
	@uv run --locked --extra docs sphinx-autobuild \
		docs docs/_build/html \
		--watch src \
		--host 0.0.0.0 --port 8020

docs-clean:
	@rm -rf docs/_build

docs-coverage:
	@uv run --locked --extra docs sphinx-build -b coverage docs docs/_build/coverage

build:
	@uv build
