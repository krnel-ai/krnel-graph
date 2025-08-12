
ONEPASSWORD_PYPI_TOKEN ?= "PyPI krnel API token"

.PHONY: test docs docs-serve docs-clean docs-autobuild publish

define get_pypi_token
	$(shell \
		if [ -n "$$PYPI_TOKEN" ]; then \
			echo "$$PYPI_TOKEN"; \
		elif [ -f ~/.pypi/token ]; then \
			cat ~/.pypi/token; \
		elif command -v op >/dev/null 2>&1; then \
			op item get $(ONEPASSWORD_PYPI_TOKEN) --fields label=credential --reveal; \
		fi)
endef

test:
	@uv run --all-extras pytest -v
test-cov:
	@uv run --all-extras pytest -v \
		--cov=src/krnel \
		--cov-report=term \
		--cov-report=xml

docs:
	@uv run --extra docs sphinx-build -b html docs docs/_build/html

docs-clean:
	@rm -rf docs/_build docs/api

docs-autobuild:
	@uv run --extra docs sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8000

publish: test
	@uv version --bump patch
	@uv build
	@uv publish \
	    --username __token__ \
		--password $(call get_pypi_token)