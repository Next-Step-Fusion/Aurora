"""Local pytest configuration for the Aurora test suite.

These tests are deliberately NOT wired into CI -- .github/workflows/tests.yml
runs example scripts, not pytest. Run them by hand:

    python -m pytest tests/ -v

Optional figures (off by default), written in the same style as
tests/test_with_omfit.py -- current run solid, stored baseline dashed:

    python -m pytest tests/test_regression.py --plot
    python -m pytest tests/test_regression.py --plot --plot-dir=some/where

Figures are produced *before* the comparisons run, so a failing test still
leaves the picture showing what moved.

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
    "regression_plots.py",
]


def pytest_addoption(parser):
    group = parser.getgroup("aurora regression")
    group.addoption(
        "--plot", action="store_true", default=False,
        help="write diagnostic figures and gifs for the regression tests "
             "(current run vs stored baseline)",
    )
    group.addoption(
        "--plot-dir", action="store", default="outputs", metavar="DIR",
        help="where those figures go (default: outputs/)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "regression: baseline-locked regression test; run manually, "
        "regenerate baselines with `python tests/test_regression.py --regenerate`",
    )
    # hand the options to the test module, which is also runnable as a script
    try:
        import test_regression
    except ImportError:
        return
    if config.getoption("--plot"):
        test_regression.PLOT_DIR = config.getoption("--plot-dir")
