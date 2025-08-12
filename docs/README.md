# Documentation

This directory contains the Sphinx documentation for krnel.

```bash
# Build HTML documentation
make docs

# Clean build artifacts
make docs-clean

# Auto-rebuild on file changes (great for development)
make docs-autobuild
```

### Local Development

For active documentation development, use:

```bash
make docs-autobuild
```

This will:
- Build the docs
- Start a local server at http://localhost:8000
- Automatically rebuild when you change files
- Live-reload the browser