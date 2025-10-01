# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "krnel-graph"
copyright = "2025, Kimberly Wilber, Peyman Faratin"
author = "Kimberly Wilber, Peyman Faratin"
release = version("krnel-graph")
version = version("krnel-graph")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoclasstoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.inheritance_diagram",
    # "autoapi.extension",
    # "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
    # "sphinx_autodoc_annotation",
    # "sphinx_autodoc_typehints",
    # "sphinx_autodoc_napoleon_typehints",
    "sphinxcontrib.mermaid",
]

autodoc_default_options = {
    "members": True,
    "special-members": False,
    "private-members": False,
    "inherited-members": False,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Type annotations
autodoc_typehints = "both"

autosummary_generate = True

# Napoleon configuration for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = {
    "krnel.graph.op_spec.OpSpecT": "krnel.graph.op_spec.OpSpec",
}
napoleon_attr_annotations = True

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# html_theme = "furo"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Create _static directory if it doesn't exist
if not os.path.exists("_static"):
    os.makedirs("_static")
