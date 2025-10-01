.PHONY: test docs docs-serve docs-clean docs-autobuild

test:
	@uv run --all-extras pytest -v

test-lowest-deps:
    # Test that using no extras works as intended:
	@uv run --isolated --exact --python=3.10 --resolution=lowest-direct python -c 'import krnel; print("Minimal import success")'
    # Test with lowest possible versions of direct dependencies:
	@uv run --isolated --exact --all-extras --python=3.10 --resolution=lowest-direct pytest -v

test-cov:
	@uv run --all-extras pytest -v \
		--cov=src/krnel \
		--cov-report=term \
		--cov-report=xml

docs:
	@uv run --extra docs sphinx-build -b html docs docs/_build/html

docs-autobuild:
	@uv run --extra docs sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8020

docs-clean:
	@rm -rf docs/_build docs/api

build:
	@uv build
