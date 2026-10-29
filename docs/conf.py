"""Sphinx configuration for crtm-py.

The user guide is Chinese. Docstrings stay English. There is no gallery:
the forward model does not plot, and the runnable example lives in the guide.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "crtm-py"
author = "crtm-py contributors"
release = "0.1.1"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 2,
    "show_nav_level": 2,
    "collapse_navigation": False,
}
html_title = "crtm-py"
html_static_path = ["_static"]
master_doc = "index"
pygments_style = "sphinx"
nitpicky = False
