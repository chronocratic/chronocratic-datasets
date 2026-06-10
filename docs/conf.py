# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# Add src/ to the path so autodoc can import chronocratic.datasets
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronocratic.datasets import __version__  # noqa: E402

project = "chronocratic-datasets"
html_title = "chronocratic-datasets"
copyright = "2026-Present, The Chronocratic Developers"
author = "The Chronocratic Developers"
release = __version__

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
autodoc_default_options = {
    "member-order": "bysource",
}
suppress_warnings = [
    "efifo",
]
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]


def skip_enum_member(app, what, name, obj, skip, options):
    """Skip raw enum values (AS_DEFINED = 'as_defined'). Attributes section in class docstring already documents these."""
    if what == "attribute" and hasattr(obj, "_name_") and hasattr(obj, "_value_"):
        return True
    return None


def setup(app):
    """Register autodoc hooks."""
    app.connect("autodoc-skip-member", skip_enum_member)
