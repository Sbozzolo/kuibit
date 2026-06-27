# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "kuibit"
copyright = "2020-2025, Gabriele Bozzola"
author = "Gabriele Bozzola"

# The full version, including alpha/beta/rc tags. Derived from the installed
# package metadata so it never goes stale when the version is bumped (kuibit is
# always installed during the documentation build).
try:
    release = _pkg_version("kuibit")
except PackageNotFoundError:
    release = "unknown"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.citations",
]

autosectionlabel_prefix_document = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "bizstyle"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

html_logo = "../logo.png"

html_show_sourcelink = False

html_sidebars = {
    "**": [
        "localtoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
        "versions.html",
    ],
}

# Versions shown in the "Versions" sidebar (docs/_templates/versions.html). Each
# entry must correspond to a directory that actually exists on the gh-pages
# branch, otherwise the link 404s. The CI workflows derive this list from the
# directories present on gh-pages and pass it in via KUIBIT_DOC_VERSIONS (a
# comma-separated list) so it can never drift out of sync. The fallback below is
# only used for local builds and lists the versions currently deployed.
_doc_versions = os.environ.get("KUIBIT_DOC_VERSIONS")
if _doc_versions:
    versions = [v.strip() for v in _doc_versions.split(",") if v.strip()]
else:
    versions = ["1.3.6", "1.4.0", "1.5.0", "1.6.1"]

# Absolute path of the documentation site root, used by the "Versions" sidebar
# (docs/_templates/versions.html) to build cross-version links. It must be
# absolute because each version is an independent Sphinx build that is unaware
# it will be served from a <version>/ subdirectory; a relative link would
# otherwise resolve as <version>/<other-version>/... . Defaults to the GitHub
# Pages project path and is overridable for forks/custom hosting.
doc_base_path = os.environ.get("KUIBIT_DOC_BASE_PATH", "/kuibit/")
if not doc_base_path.endswith("/"):
    doc_base_path += "/"

html_context = {"versions": versions, "doc_base_path": doc_base_path}

html_theme_options = {
    "maincolor": "#228B22",
}

citations_ads_token = os.environ["ADS_API"]
citations_bibcode_list = ["2021JOSS....6.3099B"]
