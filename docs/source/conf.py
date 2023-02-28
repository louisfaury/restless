# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'restless'
copyright = '2023, Louis Faury'
author = 'Louis Faury'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'recommonmark',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax'
]
source_suffix = [".rst", ".md"]


## To include README.md -- from https://www.lieret.net/2021/05/20/include-readme-sphinx/
import pathlib

# The readme that already exists
readme_path = pathlib.Path(__file__).parent.resolve().parent.resolve().parent / "README.md"
# We copy a modified version here
readme_target = pathlib.Path(__file__).parent / "readme.md"

with readme_target.open("w") as outf:
    # Change the title to "Readme"
    lines = []
    for line in readme_path.read_text().split("\n"):
        if line.startswith("# "):
            # Skip title, because we now use "Readme"
            # Could also simply exclude first line for the same effect
            continue
        lines.append(line)
    outf.write("\n".join(lines))
####

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_flags = ['members', 'inherited-members']
autodoc_default_options = {'special-members': '__call__'}
autodoc_member_order = 'alphabetical'

# Describe class content
autoclass_content = 'both'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
