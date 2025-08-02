
ONEPASSWORD_PYPI_TOKEN ?= "PyPI krnel API token"

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
	@uv run pytest

test-verbose:
	@uv run pytest -v

test-coverage:
	@uv run pytest --cov=src/krnel --cov-report=html --cov-report=term

publish:
	@uv version --bump patch
	@uv build
	@uv publish \
	    --username __token__ \
		--password $(call get_pypi_token)