"""Local pytest configuration for the Aurora test suite.

These tests are deliberately NOT wired into CI -- .github/workflows/tests.yml
runs example scripts, not pytest. Run them by hand:

    python -m pytest tests/ -v

``test_core_impurity.py`` and ``test_with_omfit.py`` are *scripts*, not pytest
modules: they do their work at import time (full simulations, figures, a gif),
and ``test_with_omfit.py`` additionally needs OMFIT. Their filenames match
pytest's discovery pattern, so they are excluded here -- otherwise
``pytest tests/`` would silently run them. Execute them directly instead:

    python tests/test_with_omfit.py
"""

collect_ignore = [
    "test_core_impurity.py",
    "test_with_omfit.py",
    "helpers.py",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "regression: baseline-locked regression test; run manually, "
        "regenerate baselines with `python tests/test_regression.py --regenerate`",
    )
