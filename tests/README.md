# Aurora tests

This directory holds one pytest suite plus two standalone scripts.

**None of it is wired into CI** — `.github/workflows/tests.yml` runs example scripts, not pytest.
Run everything here by hand.

```bash
python -m pytest tests/ -v
```

Requires an Aurora install on **Python ≥ 3.11 / numpy ≥ 2** (see `TASK4_VERSION_UPGRADE.md`).
The repo's in-tree `.venv` is Python 3.8 / numpy 1.24 and is below that floor — it cannot run the
suite. It is still the only environment that can run `test_with_omfit.py`, which needs OMFIT.

---

## 1. `test_regression.py` — the baseline-locked regression suite

27 tests. Freezes the solver's numerical behaviour so refactoring can be shown not to change the
physics. **The physics case is defined entirely inside the file** — no HDF5, no geqdsk, no OMFIT —
which keeps the baselines independent of the input-data pipeline.

What is tracked, for **C** and **W**:

* `nz(t, z, r)` — impurity density per charge state [cm⁻³]
* `line_rad(t, z, r)` — line radiation, ADAS `plt` [W cm⁻³]
* `cont_rad(t, z, r)` — continuum (recombination + bremsstrahlung), ADAS `prb` [W cm⁻³]
* volume-integrated inventories and radiated powers vs time
* the radial grid itself
* FACIT neoclassical `Dz` / `Vconv` per charge state

### The tests

| Test | Cases | What it asserts |
|---|---|---|
| `test_radial_grid` | C, W | The grid itself must not move — everything else is defined on it |
| `test_impurity_density` | C, W | `nz(t,z,r)` at 5 time slices, full radial/charge resolution |
| `test_line_radiation` | C, W | Line radiation per charge state (ADAS `plt`) |
| `test_continuum_radiation` | C, W | Continuum radiation per charge state (ADAS `prb`) |
| `test_inventory_time_trace` | C, W | Volume-integrated confined inventory per charge state, every step |
| `test_radiated_power_time_trace` | C, W | Volume-integrated line and continuum power, every step |
| `test_external_grid_path` | C, W | Same physics through namelist `rvol_grid` + `rhop_grid` |
| `test_physical_sanity` | C, W | Invariants that hold regardless of the baseline (finite, non-negative, monotonic grid) |
| `test_negative_density_undershoot` | C, W | The scheme's positivity error, frozen per charge state |
| `test_stepped_restart` | C | The integrated-modelling coupling pattern: restart each step feeding the previous `nz` as `nz_init` |
| `test_facit_coefficients` | C, W × `rotation_model` 0, 1, 2 | Neoclassical `Dz` and `Vconv` per charge state |
| `test_facit_multi_species` | C+W | Two intrinsic impurities coupled through a shared `Zeff(r)`, each transported with its own FACIT D/V on a flat anomalous background |
| `test_multi_species_differs_from_single` | C+W | Guard on the above: the shared `Zeff` must actually change the answer versus a single-species run |

`rotation_model=2` is worth singling out: it integrates with `np.trapezoid`, which replaced the
`np.trapz` that numpy 2 removed. These two tests are what showed that conversion moved nothing.

Two run patterns are covered, mirroring `examples/core_impurity.py`:
`single` (one `run_aurora` call over the whole window) and `stepped` (restart per step, which
exercises state hand-off — the pattern an integrated model actually uses).

### Baselines

Stored as `baselines/baseline_*.npz`, one per case. Each file records its own provenance in a
`_meta` entry:

```bash
python -c "import numpy as np, json, sys; \
  print(json.dumps(json.loads(str(np.load('tests/baselines/baseline_C.npz')['_meta'])), indent=2))"
```

The current set was generated on **Python 3.11.13 / numpy 2.4.6 / scipy 1.17.1**.
`old_baseline/` (untracked) holds the previous **Python 3.8.20 / numpy 1.24.4 / scipy 1.10.1** set;
the two agree to 3.6e-9 of peak across all 126 arrays.

Regenerate **only** when a change in output is intended and has been reviewed — that review is
what makes the test meaningful:

```bash
python tests/test_regression.py --regenerate
```

Running the file without `--regenerate` prints its docstring and exits.

---

## 2. How to run it

```bash
# the whole suite
python -m pytest tests/ -v

# just the regression file
python -m pytest tests/test_regression.py -v

# one test, or one case
python -m pytest tests/test_regression.py::test_impurity_density -v
python -m pytest "tests/test_regression.py::test_facit_coefficients[2-W]" -v

# everything touching FACIT / everything for W
python -m pytest tests/test_regression.py -k facit -v
python -m pytest tests/test_regression.py -k "W" -v
```

Each physics case is run once per session and cached, so selecting a subset does not re-run the
solver for every test.

### Optional flags

| Flag | Default | What it does |
|---|---|---|
| `--plot` | off | Write diagnostic figures and gifs: **current run solid, stored baseline dashed** |
| `--plot-dir=DIR` | `outputs` | Where those figures go |

```bash
python -m pytest tests/test_regression.py --plot
python -m pytest tests/test_regression.py --plot --plot-dir=some/where
```

Figures are produced *before* the comparisons run, so a failing test still leaves the picture
showing what moved. Plotting can never fail a test — problems are reported and swallowed. The gif
needs `imageio`; if it is missing you get a skip line, not an error.

Files written per case `<name>`: `nz_<name>.png`, `prad_<name>.png`, `traces_<name>.png`,
`nz_evolution_<name>.gif`, `facit_<name>.png`, and `multi_species.png`.

`--plot` / `--plot-dir` are pytest options, so they do not exist in script mode — there, pass
`--plot [DIR]` positionally. The `AURORA_PLOT_DIR` / `AURORA_PLOT` environment variables below are
read at import and therefore work in **both** modes, which makes them the portable way to turn
figures on.

### Environment variables

| Variable | Default | What it does |
|---|---|---|
| `AURORA_REGRESSION_RTOL` | `1e-9` | Relative comparison tolerance |
| `AURORA_REGRESSION_ATOL_FRAC` | `1e-11` | Absolute floor, as a fraction of each array's own peak |
| `AURORA_PLOT_DIR` | unset | Figure directory. Setting it turns figures on, under pytest and in script mode alike — no `--plot` needed |
| `AURORA_PLOT` | unset | If set to anything, turns figures on and defaults the directory to `outputs` |
| `AURORA_ADAS_DIR` | unset | Where Aurora looks for / downloads ADAS files. Set it to avoid writing into `site-packages` |

The solver is bit-deterministic within one environment — the current run reproduces the stored
baselines *exactly*, so the suite still passes at `AURORA_REGRESSION_RTOL=1e-20`. That is why the
default is deliberately tight. Relax it for a **cross-version** comparison — different numpy/scipy/BLAS/compiler
combinations shift the last few digits:

```bash
AURORA_REGRESSION_RTOL=1e-7 python -m pytest tests/test_regression.py -v
```

`ATOL_FRAC` is scaled by each array's own peak so that the vast almost-zero regions of `nz` and
`Prad` do not dominate the comparison.

> **First run needs network.** The case reads ADAS `acd`/`scd`/`plt`/`prb` files for C and W;
> Aurora downloads any that are missing from `open.adas.ac.uk`. Set `AURORA_ADAS_DIR` to a
> writable directory to cache them outside the installed package.

---

## 3. The two standalone scripts

These are **scripts, not pytest modules** — they do their work at import time (full simulations,
figures, a gif). Their filenames match pytest's discovery pattern, so `conftest.py` excludes them
from collection; otherwise `pytest tests/` would silently run them. Execute them directly:

| Script | Command | Extra requirements |
|---|---|---|
| `test_core_impurity.py` | `python tests/test_core_impurity.py` | `h5py`, `imageio`, `freeqdsk`; reads `data/centaur.h5` and `data/centaur.geqdsk` |
| `test_with_omfit.py` | `python tests/test_with_omfit.py` | **OMFIT** (`omfit_classes`), `imageio`; reads `data/centaur.geqdsk` |

`test_with_omfit.py` is developer-only and the one place in the repo that still imports OMFIT on
purpose. It runs the same physics case twice — once through the OMFIT geqdsk path and once through
the OMFIT-free namelist path — and compares them. It is the reference the regression suite above
was derived from, kept so the equivalence check can still be reproduced in a legacy environment.

Note `test_core_impurity.py` reads its g-file with **`freeqdsk`**, not OMFIT — a useful precedent
for the OMFIT-free equilibrium interface that `TASK3_OMFIT_REMOVAL.md` §6 leaves open.

---

## 4. Supporting files

| File | Role |
|---|---|
| `conftest.py` | Registers `--plot` / `--plot-dir`, the `regression` marker, and the collection ignores |
| `helpers.py` | Shared plot style and comparison helpers (run-a-case, common grid, volume integral, profile stats). Free of import-time side effects |
| `regression_plots.py` | All figure and gif generation for `--plot` |
| `baselines/` | The stored `.npz` baselines the suite compares against |
| `old_baseline/` | Untracked archive of the pre-upgrade py3.8 baselines |
| `data/` | `centaur.h5` (32 MB transport-solver state) and `centaur.geqdsk`, used only by the two scripts |
