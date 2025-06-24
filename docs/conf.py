# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Segma Vision Pro'
copyright = '2025, DOUAE JAMAI'
author = 'DOUAE JAMAI'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Configuration pour éviter les erreurs
master_doc = 'index'