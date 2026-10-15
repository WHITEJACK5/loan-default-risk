"""Single source of truth for the package version.

``pyproject.toml`` derives its version from ``__version__`` here, and the API
reads it from the same place. The version used to be written out in four
separate files -- ``pyproject.toml``, ``README.md`` and two literals in
``src/api/main.py`` -- so it drifted twice (D5), and PyPI metadata disagreed with
the FastAPI ``/health`` payload and the introspected version.
"""

__version__ = "2.1.0"
