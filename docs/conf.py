# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# Add src/ to the path so autodoc can import chronocratic.datasets
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "chronocratic-datasets"
copyright = "2024-2026, The Chronocratic Developers"
author = "The Chronocratic Developers"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 3,
    "show_toc_level": 2,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

napoleon_use_google_style = True
