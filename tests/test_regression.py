"""Baseline-locked regression tests for the Aurora impurity-transport solver.

Purpose
-------
Freeze the current numerical behaviour of the solver so that the planned
refactor (removing OMFIT, Python >= 3.11, numpy >= 2, module pruning, external
radial grids) can be shown not to change the physics.

What is tracked, for **C** and **W**:
  * ``nz(t, z, r)``        impurity density per charge state      [cm^-3]
  * ``line_rad(t, z, r)``  line radiation, ADAS "plt"             [W cm^-3]
  * ``cont_rad(t, z, r)``  continuum radiation, ADAS "prb"        [W cm^-3]
  * volume-integrated inventories and radiated powers vs time
  * the radial grid itself

Two run patterns are covered, mirroring ``examples/core_impurity.py``:
  1. ``single``   -- one ``run_aurora`` call over the whole window
  2. ``stepped``  -- restart each step feeding the previous ``nz`` as
                     ``nz_init``; this is the integrated-modelling coupling
                     pattern, and it exercises state hand-off

FACIT neoclassical transport is covered too, in both roles it plays:
  * the coefficients themselves, ``Dz``/``Vconv`` per charge state, for
    ``rotation_model`` 0, 1 and 2
  * a two-species run with **C and W together as intrinsic impurities**,
    coupled through a shared Zeff(r) built from both species' densities, each
    then transported with its own FACIT D/V on a flat anomalous background --
    the chain an integrated model actually uses

``rotation_model=2`` matters especially: it used to call ``np.trapz``, which
numpy 2 removed. That call is now ``np.trapezoid`` and this baseline is what
showed the conversion did not move the coefficients.

The physics case is defined **entirely inside this file** -- no HDF5, no geqdsk,
no OMFIT. That keeps the baselines immune to fixes in the input data pipeline
and lets the tests run before and after the OMFIT removal.

NOT wired into CI -- run manually
--------------------------------
    python -m pytest tests/test_regression.py -v

Regenerate the baselines (only when a change in output is intended and has been
reviewed -- this is what makes the test meaningful):

    python tests/test_regression.py --regenerate

Tolerance can be relaxed for a cross-version comparison, e.g. after moving to
numpy 2:

    AURORA_REGRESSION_RTOL=1e-6 python -m pytest tests/test_regression.py -v

Optional figures -- current run solid, stored baseline dashed, exactly as in
tests/test_with_omfit.py. Off by default; written to ``outputs/``:

    python -m pytest tests/test_regression.py --plot
    python -m pytest tests/test_regression.py --plot --plot-dir=some/where
"""

import json
import os
import platform
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(HERE, "baselines")

# Comparison tolerance. The solver is bit-deterministic within one environment,
# so the default is tight; relax via the environment variable when comparing
# across numpy/scipy/Python versions.
RTOL = float(os.environ.get("AURORA_REGRESSION_RTOL", "1e-9"))
# absolute floor, as a fraction of each array's own peak, so that the vast
# almost-zero regions of nz/Prad do not dominate the comparison
ATOL_FRAC = float(os.environ.get("AURORA_REGRESSION_ATOL_FRAC", "1e-11"))

N_SLICES = 5          # time slices stored at full radial/charge resolution
IMPURITIES = ["C", "W"]

# Set by conftest's --plot / --plot-dir, or by AURORA_PLOT_DIR when this file is
# run as a script. None means "no figures". Default directory is outputs/.
PLOT_DIR = os.environ.get("AURORA_PLOT_DIR") or (
    "outputs" if os.environ.get("AURORA_PLOT") else None)

pytestmark = pytest.mark.regression


# ---------------------------------------------------------------------------
# The frozen physics case
# ---------------------------------------------------------------------------
# Magnitudes mirror the Centaur reference case (R0 ~ 2.12 m, a_vol ~ 0.8 m,
# ne(0) ~ 3.8e14 cm^-3, Te(0) ~ 24.5 keV) but every number here is explicit, so
# the baselines depend on nothing outside this file.
CASE = dict(
    Raxis_cm=212.072,
    Baxis=10.9,
    rvol_lcfs=79.799,
    bound_sep=2.0,
    lim_sep=1.0,
    K=6.0,
    dr_0=0.3,
    dr_1=0.05,
    source_rate=1e21,
    D_cm2s=1e4,
    V_cms=0.0,
    t_end=0.1,
    dt=1e-3,
    n_prof=100,
)


def _profiles(n=None):
    """Analytic kinetic profiles on a rho_pol grid."""
    n = n or CASE["n_prof"]
    rhop = np.linspace(0.0, 1.0, n)
    ne = (3.8e14 - 0.14e14) * (1 - rhop**2) ** 0.5 + 0.14e14   # cm^-3
    Te = (2.45e4 - 10.0) * (1 - rhop**2) ** 1.5 + 10.0         # eV
    Ti = Te.copy()
    n0 = 1e10 * np.exp((rhop - 1.0) / 0.02)                    # cm^-3
    return rhop, ne, Te, Ti, n0


def _external_grid():
    """A deliberately non-analytic radial grid + rho_pol mapping.

    Exercises the external-grid path (``rvol_grid`` / ``rhop_grid`` in the
    namelist -> ``grids_utils.grid_from_rvol``). The mapping
    ``rho_pol = (r/a)**0.85`` is synthetic but monotone and clearly distinct
    from the rho_vol fallback ``r/a``, so a regression that silently drops
    ``rhop_grid`` will show up here.
    """
    import aurora

    nml = {k: CASE[k] for k in ("K", "dr_0", "dr_1", "rvol_lcfs", "bound_sep",
                                "lim_sep")}
    rvol = aurora.grids_utils.create_radial_grid(nml)[0]
    x = rvol / CASE["rvol_lcfs"]
    rhop = np.where(x <= 1.0, x**0.85, 1.0 + 0.85 * (x - 1.0))
    rhop[0] = 0.0
    return rvol, rhop


def _namelist(imp, external_grid=False):
    import aurora

    rhop, ne, Te, Ti, n0 = _profiles()
    nml = aurora.load_default_namelist()
    nml["imp"] = imp
    nml["Raxis_cm"] = CASE["Raxis_cm"]
    nml["Baxis"] = CASE["Baxis"]
    nml["rvol_lcfs"] = CASE["rvol_lcfs"]
    nml["bound_sep"] = CASE["bound_sep"]
    nml["lim_sep"] = CASE["lim_sep"]
    nml["K"] = CASE["K"]
    nml["dr_0"] = CASE["dr_0"]
    nml["dr_1"] = CASE["dr_1"]
    nml["source_type"] = "const"
    nml["source_rate"] = CASE["source_rate"]
    nml["LBO"] = None
    for key, vals in (("ne", ne), ("Te", Te), ("Ti", Ti), ("n0", n0)):
        nml["kin_profs"][key]["rhop"] = rhop
        nml["kin_profs"][key]["vals"] = vals
    nml["kin_profs"]["Te"]["decay"] = [1.0]
    nml["kin_profs"]["Ti"]["decay"] = [1.0]
    if external_grid:
        rvol_ext, rhop_ext = _external_grid()
        nml["rvol_grid"] = rvol_ext
        nml["rhop_grid"] = rhop_ext
    return nml


def _timing(t0, t1):
    return {
        "dt_increase": np.array([1.0, 1.0]),
        "dt_start": np.array([CASE["dt"], CASE["dt"]]),
        "steps_per_cycle": np.array([1, 1]),
        "times": np.array([t0, t1]),
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def _summarise(asim, nz, imp):
    """Reduce a run to the arrays that get frozen.

    ``nz`` arrives as (nr, nZ, nt); everything stored is (nt, nZ, nr) so the
    time axis leads, matching ``compute_rad``.
    """
    import aurora

    nz_t = np.ascontiguousarray(nz.transpose(2, 1, 0))          # (nt, nZ, nr)
    rad = aurora.radiation.compute_rad(
        imp, nz_t, asim._ne, asim._Te, prad_flag=True,
        thermal_cx_rad_flag=False, spectral_brem_flag=False, sxr_flag=False,
    )
    line_rad, cont_rad = rad["line_rad"], rad["cont_rad"]

    nt = nz_t.shape[0]
    isl = np.unique(np.linspace(0, nt - 1, N_SLICES).astype(int))

    def vint(prof2d):                     # (nt, nr) -> (nt,)
        return aurora.grids_utils.vol_int(
            prof2d, asim.rvol_grid, asim.pro_grid, asim.Raxis_cm,
            rvol_max=asim.rvol_lcfs)

    # confined inventory per charge state and time -> (nt, nZ)
    n_conf = np.stack([vint(nz_t[:, z, :]) for z in range(nz_t.shape[1])], axis=1)

    # negative-density undershoot: this scheme does not guarantee positivity
    # (examples/core_impurity.py clips with np.maximum(nz, 0)). Freeze it so a
    # refactor that makes it worse is caught.
    pk_z = np.nanmax(np.abs(nz_t), axis=(0, 2))
    mn_z = np.nanmin(nz_t, axis=(0, 2))
    undershoot = mn_z / np.maximum(pk_z, 1e-300)
    # cont_rad = nz[:, 1:] * prb, so it inherits the nz undershoot directly
    # (line_rad is clamped to >= 1e-60 inside compute_rad and stays positive)
    cont_undershoot = np.array(
        np.nanmin(cont_rad) / max(np.nanmax(np.abs(cont_rad)), 1e-300))

    return dict(
        undershoot=undershoot,
        cont_undershoot=cont_undershoot,
        neg_fraction=np.array((nz_t < 0).mean()),
        t_slices=asim.time_out[isl],
        slice_idx=isl,
        rhop=asim.rhop_grid,
        rvol=asim.rvol_grid,
        pro=asim.pro_grid,
        qpr=asim.qpr_grid,
        time=asim.time_out,
        nz=nz_t[isl],
        line_rad=line_rad[isl],
        cont_rad=cont_rad[isl],
        n_conf=n_conf,
        p_line=vint(line_rad.sum(1)),
        p_cont=vint(cont_rad.sum(1)),
    )


def run_single(imp, external_grid=False):
    """One continuous ``run_aurora`` call over the whole window."""
    import aurora

    nml = _namelist(imp, external_grid=external_grid)
    nml["timing"] = _timing(0.0, CASE["t_end"])
    asim = aurora.aurora_sim(nml)
    nr = len(asim.rvol_grid)
    D_z = CASE["D_cm2s"] * np.ones(nr)
    V_z = CASE["V_cms"] * np.ones(nr)
    out = asim.run_aurora(D_z, V_z)
    return _summarise(asim, out["nz"], imp)


def run_stepped(imp, n_steps=20):
    """Restart each step with the previous ``nz`` as ``nz_init``.

    This is the coupling pattern of ``examples/core_impurity.py``: the sim is
    rebuilt every step and only the impurity density is carried forward, so the
    edge reservoirs restart each time. It is frozen separately from the
    continuous run precisely because it is *not* equivalent to it.
    """
    import aurora

    nml = _namelist(imp)
    dt_step = CASE["t_end"] / n_steps
    nz = None
    asim = None
    for it in range(n_steps):
        nml["timing"] = {
            "dt_increase": np.array([1.0, 1.0]),
            "dt_start": np.array([CASE["dt"], CASE["dt"]]),
            "steps_per_cycle": np.array([1, 1]),
            "times": np.array([it * dt_step, (it + 1) * dt_step]),
        }
        asim = aurora.aurora_sim(nml)
        if nz is None:
            nz = np.zeros((len(asim.rvol_grid), asim.Z_imp + 1))
        D_z = CASE["D_cm2s"] * np.ones(len(asim.rvol_grid))
        V_z = CASE["V_cms"] * np.ones(len(asim.rvol_grid))
        out = asim.run_aurora(D_z, V_z, nz_init=nz)
        nz = np.maximum(out["nz"][:, :, -1], 0.0)
    return dict(nz_final=nz, rhop=asim.rhop_grid, rvol=asim.rvol_grid)


# ---------------------------------------------------------------------------
# FACIT neoclassical transport
# ---------------------------------------------------------------------------
# FACIT works in SI (m, m^-3, m^2/s, m/s); Aurora is CGS. Conversions are
# Dz * 1e4 -> cm^2/s and Vconv * 1e2 -> cm/s.
FACIT_CASE = dict(
    Machi=0.25,          # main-ion Mach number (only used by rotation_model 1/2)
    Zeff=1.5,
    Te_Ti=1.0,
    q0=1.0, q_a=2.0,     # qmag = q0 + q_a * roa**2
    # FACIT divides by the impurity density, and low charge states of a light
    # impurity are identically zero across the core. Floor Nimp at a trace
    # level of ne so the coefficients stay finite -- facit_basic.py makes the
    # same move with its `c_imp` trace concentration.
    trace_floor=1e-10,
    D_anom_cm2s=1e4,     # flat anomalous background the neoclassical part adds to
    V_anom_cms=0.0,
)
ROTATION_MODELS = [0, 1, 2]


def _facit_inputs(asim, nz_final):
    """Assemble FACIT's SI inputs from an Aurora state.

    ``roa`` is taken as ``rvol / rvol_lcfs``. For a real equilibrium the
    mid-plane r/a and the volume radius differ; here the case is synthetic and
    the choice only has to be fixed, not exact.
    """
    m = (asim.rvol_grid <= asim.rvol_lcfs) & (asim.rvol_grid > 0.0)
    roa = asim.rvol_grid[m] / asim.rvol_lcfs
    amin_m = asim.rvol_lcfs / 100.0
    R0_m = asim.Raxis_cm / 100.0
    r_m = roa * amin_m
    Ti = asim._Te[-1][m]                       # eV  (Ti = Te in this case)
    Ni = asim._ne[-1][m] * 1e6                 # cm^-3 -> m^-3
    return dict(
        mask=m, roa=roa, r_m=r_m, amin_m=amin_m, R0_m=R0_m,
        Ti=Ti, Ni=Ni,
        gradTi=np.gradient(Ti, r_m), gradNi=np.gradient(Ni, r_m),
        qmag=FACIT_CASE["q0"] + FACIT_CASE["q_a"] * roa**2,
        floor=FACIT_CASE["trace_floor"] * Ni,
        nz=nz_final,
    )


def _facit_coefficients(asim, inp, rotation_model):
    """Dz [m^2/s] and Vconv [m/s] per charge state on the core grid."""
    import aurora

    nZ = asim.Z_imp + 1
    Dz = np.zeros((len(inp["roa"]), nZ))
    Vz = np.zeros_like(Dz)
    Machi = FACIT_CASE["Machi"] if rotation_model else 0.0
    for z in range(1, nZ):
        Nz = np.maximum(inp["nz"][inp["mask"], z] * 1e6, inp["floor"])
        fct = aurora.FACIT(
            inp["roa"], z, asim.A_imp, 1, 2,
            inp["Ti"], inp["Ni"], Nz, Machi, FACIT_CASE["Zeff"],
            inp["gradTi"], inp["gradNi"], np.gradient(Nz, inp["r_m"]),
            inp["amin_m"] / inp["R0_m"], CASE["Baxis"], inp["R0_m"], inp["qmag"],
            rotation_model=rotation_model, Te_Ti=FACIT_CASE["Te_Ti"],
        )
        Dz[:, z] = fct.Dz
        Vz[:, z] = fct.Vconv
    return Dz, Vz


def _flat_run(imp):
    """Stage 1: a flat-transport run, used only to give FACIT an nz profile."""
    import aurora

    nml = _namelist(imp)
    nml["timing"] = _timing(0.0, CASE["t_end"])
    asim = aurora.aurora_sim(nml)
    nr = len(asim.rvol_grid)
    out = asim.run_aurora(CASE["D_cm2s"] * np.ones(nr),
                          CASE["V_cms"] * np.ones(nr))
    return asim, np.maximum(out["nz"][:, :, -1], 0.0)


def run_facit(imp, rotation_model):
    """Freeze the FACIT coefficients for a fixed plasma state."""
    asim, nz = _flat_run(imp)
    inp = _facit_inputs(asim, nz)
    Dz, Vz = _facit_coefficients(asim, inp, rotation_model)
    return dict(roa=inp["roa"], Dz=Dz, Vconv=Vz,
                Dz_max=np.array(np.nanmax(Dz)),
                Vconv_min=np.array(np.nanmin(Vz)))


def _transport_with(asim, inp, Dz_si, Vz_si, imp):
    """Re-run this species with FACIT D/V added to a flat anomalous background."""
    import aurora

    nr = len(asim.rvol_grid)
    nZ = asim.Z_imp + 1
    D_z = np.full((nr, 1, nZ), FACIT_CASE["D_anom_cm2s"])
    V_z = np.full((nr, 1, nZ), FACIT_CASE["V_anom_cms"])
    D_z[inp["mask"], 0, :] += Dz_si * 1e4        # m^2/s -> cm^2/s
    V_z[inp["mask"], 0, :] += Vz_si * 1e2        # m/s   -> cm/s

    nml = _namelist(imp)
    nml["timing"] = _timing(0.0, CASE["t_end"])
    asim2 = aurora.aurora_sim(nml)
    out = asim2.run_aurora(D_z, V_z, times_DV=np.array([0.0]))
    res = _summarise(asim2, out["nz"], imp)
    res["D_z"] = D_z[:, 0, :]
    res["V_z"] = V_z[:, 0, :]
    res["Dz_si"] = Dz_si
    res["Vconv_si"] = Vz_si
    return res


def zeff_from_species(nz_by_imp, ne):
    """Zeff = 1 + sum_species sum_z n_z Z(Z-1) / ne.

    Same expression as :py:meth:`aurora.core.aurora_sim.calc_Zeff`, summed over
    species rather than over one.
    """
    dZ = {}
    for imp, nz in nz_by_imp.items():
        Z = np.arange(nz.shape[1])
        dZ[imp] = (nz * (Z * (Z - 1))[None, :]).sum(1) / ne
    return 1.0 + sum(dZ.values()), dZ


def run_multi_species(rotation_model=2):
    """C and W as two intrinsic impurities sharing one plasma background.

    Aurora evolves one species per ``aurora_sim``, so "together" means two sims
    coupled through the quantity they actually share: the effective charge.

      stage 1   flat transport for each species  -> nz_C, nz_W
      stage 2   Zeff(r) from BOTH species' densities
      stage 3   FACIT per species, using that shared Zeff(r)
      stage 4   re-run each species with its neoclassical D/V

    One Picard pass, deterministic -- not iterated to a fixed point.

    ``rotation_model=2`` for both species is deliberate: at ``rotation_model=0``
    FACIT ignores Zeff entirely (``CgeoU = 0`` in the poloidally symmetric
    limit, facit.py:344), so the species coupling under test would be a no-op.
    """
    sims, nz0 = {}, {}
    for imp in IMPURITIES:
        asim, nz = _flat_run(imp)
        sims[imp], nz0[imp] = asim, nz

    ne = sims[IMPURITIES[0]]._ne[-1]                   # shared background
    for imp in IMPURITIES[1:]:
        assert np.allclose(ne, sims[imp]._ne[-1]), "species see different ne"

    Zeff, dZ = zeff_from_species(nz0, ne)

    out = {"Zeff": Zeff, "ne": ne,
           "rhop": sims[IMPURITIES[0]].rhop_grid}
    for imp in IMPURITIES:
        out[f"dZeff_{imp}"] = dZ[imp]
    saved = FACIT_CASE["Zeff"]
    try:
        for imp in IMPURITIES:
            asim = sims[imp]
            inp = _facit_inputs(asim, nz0[imp])
            FACIT_CASE["Zeff"] = Zeff[inp["mask"]]     # shared, radially varying
            Dz_si, Vz_si = _facit_coefficients(asim, inp, rotation_model)
            res = _transport_with(asim, inp, Dz_si, Vz_si, imp)
            out[f"{imp}_roa"] = inp["roa"]
            for k in ("nz", "line_rad", "cont_rad", "n_conf", "D_z", "V_z",
                      "Dz_si", "Vconv_si", "rhop", "rvol", "t_slices"):
                out[f"{imp}_{k}"] = res[k]
    finally:
        FACIT_CASE["Zeff"] = saved
    return out


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------
def baseline_path(name):
    return os.path.join(BASELINE_DIR, f"baseline_{name}.npz")


def _meta():
    import aurora
    import scipy

    return json.dumps({
        "aurora": aurora.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "case": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                 for k, v in CASE.items()},
    }, indent=2, sort_keys=True)


def save_baseline(name, data):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    keep = {k: v for k, v in data.items() if not k.startswith("_")}
    np.savez_compressed(baseline_path(name), _meta=np.array(_meta()), **keep)
    return baseline_path(name)


def load_baseline(name):
    p = baseline_path(name)
    if not os.path.exists(p):
        pytest.skip(f"no baseline at {p} -- generate with "
                    f"`python tests/test_regression.py --regenerate`")
    return np.load(p, allow_pickle=False)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def assert_matches(ref, tst, what, rtol=None, atol=None):
    """np.allclose with a failure message that says *where* it went wrong."""
    ref = np.asarray(ref, dtype=float)
    tst = np.asarray(tst, dtype=float)
    rtol = RTOL if rtol is None else rtol
    assert ref.shape == tst.shape, (
        f"{what}: shape changed {ref.shape} -> {tst.shape}")

    peak = np.nanmax(np.abs(ref))
    atol = (ATOL_FRAC * peak if peak > 0 else 1e-30) if atol is None else atol

    diff = np.abs(tst - ref)
    tolm = atol + rtol * np.abs(ref)
    bad = diff > tolm
    if not bad.any():
        return

    i = np.unravel_index(np.argmax(diff - tolm), diff.shape)
    rel = diff[i] / max(abs(ref[i]), peak * 1e-30)
    raise AssertionError(
        f"{what}: {bad.sum()} of {bad.size} values outside "
        f"rtol={rtol:g}, atol={atol:g}\n"
        f"  worst at index {i}: baseline={ref[i]:.12e}  now={tst[i]:.12e}\n"
        f"  abs diff {diff[i]:.6e}   rel diff {rel:.6e}   "
        f"(array peak {peak:.6e})\n"
        f"  max |diff| over array = {diff.max():.6e}"
    )


# ---------------------------------------------------------------------------
# Optional figures
# ---------------------------------------------------------------------------
def _plot(kind, name, ref, now, **kw):
    """Write figures for one case, if --plot was given.

    Called *before* the assertions so that a failing comparison still leaves the
    picture. Never allowed to break a test: plotting problems are reported and
    swallowed.
    """
    if PLOT_DIR is None:
        return []
    try:
        import regression_plots as P

        out = P.outdir(PLOT_DIR)
        made = []
        if kind == "profiles":
            imp = kw["imp"]
            made.append(P.plot_profiles(name, ref, now, out, imp))
            made.append(P.plot_radiation(name, ref, now, out))
            made.append(P.plot_traces(name, ref, now, out))
            made.append(P.animate_profiles(name, ref, now, out, imp))
        elif kind == "facit":
            made.append(P.plot_facit(name, ref, now, out, kw["imp"]))
        elif kind == "multi":
            made.append(P.plot_multi_species(ref, now, out, IMPURITIES))
        print(f"\n  [plot] {name}: " + ", ".join(os.path.basename(m)
                                                 for m in made))
        return made
    except Exception as exc:                       # never fail a test on a plot
        print(f"\n  [plot] {name}: skipped ({type(exc).__name__}: {exc})")
        return []


# ---------------------------------------------------------------------------
# Cached runs -- each case is executed once per session
# ---------------------------------------------------------------------------
_CACHE = {}


def _cached(name, fn, *a, **kw):
    if name not in _CACHE:
        _CACHE[name] = fn(*a, **kw)
    return _CACHE[name]


@pytest.fixture(scope="session")
def result(request):
    imp, kind = request.param
    if kind == "single":
        return imp, _cached(imp, run_single, imp)
    if kind == "ext":
        return imp, _cached(imp + "_ext", run_single, imp, external_grid=True)
    raise ValueError(kind)


def _params(kind, imps=IMPURITIES):
    return pytest.mark.parametrize(
        "result", [pytest.param((i, kind), id=f"{i}-{kind}") for i in imps],
        indirect=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@_params("single")
def test_radial_grid(result):
    """The grid itself must not move -- everything else is defined on it."""
    imp, r = result
    b = load_baseline(imp)
    assert_matches(b["rvol"], r["rvol"], f"{imp}: rvol_grid")
    assert_matches(b["rhop"], r["rhop"], f"{imp}: rhop_grid")
    assert_matches(b["pro"], r["pro"], f"{imp}: pro_grid")
    assert_matches(b["qpr"], r["qpr"], f"{imp}: qpr_grid")
    assert_matches(b["time"], r["time"], f"{imp}: time_out")


@_params("single")
def test_impurity_density(result):
    """nz(t, z, r) at N_SLICES time slices, full radial/charge resolution."""
    imp, r = result
    b = load_baseline(imp)
    _plot("profiles", imp, b, r, imp=imp)
    assert_matches(b["t_slices"], r["t_slices"], f"{imp}: slice times")
    assert_matches(b["nz"], r["nz"], f"{imp}: nz(t,z,r)")


@_params("single")
def test_line_radiation(result):
    """Line radiation per charge state, ADAS plt."""
    imp, r = result
    b = load_baseline(imp)
    assert_matches(b["line_rad"], r["line_rad"], f"{imp}: line_rad(t,z,r)")


@_params("single")
def test_continuum_radiation(result):
    """Recombination + bremsstrahlung continuum, ADAS prb."""
    imp, r = result
    b = load_baseline(imp)
    assert_matches(b["cont_rad"], r["cont_rad"], f"{imp}: cont_rad(t,z,r)")


@_params("single")
def test_inventory_time_trace(result):
    """Volume-integrated confined inventory per charge state, every time step."""
    imp, r = result
    b = load_baseline(imp)
    assert_matches(b["n_conf"], r["n_conf"], f"{imp}: confined inventory(t,z)")


@_params("single")
def test_radiated_power_time_trace(result):
    """Volume-integrated line and continuum power, every time step."""
    imp, r = result
    b = load_baseline(imp)
    assert_matches(b["p_line"], r["p_line"], f"{imp}: P_line(t)")
    assert_matches(b["p_cont"], r["p_cont"], f"{imp}: P_cont(t)")


@_params("ext")
def test_external_grid_path(result):
    """Same physics through namelist rvol_grid + rhop_grid.

    Guards the external-grid entry point: a regression that ignores
    ``rhop_grid`` collapses rho_pol onto rho_vol and this fails.
    """
    imp, r = result
    b = load_baseline(imp + "_ext")
    _plot("profiles", imp + "_ext", b, r, imp=imp)
    assert_matches(b["rvol"], r["rvol"], f"{imp} ext: rvol_grid")
    assert_matches(b["rhop"], r["rhop"], f"{imp} ext: rhop_grid")
    assert_matches(b["nz"], r["nz"], f"{imp} ext: nz(t,z,r)")
    assert_matches(b["line_rad"], r["line_rad"], f"{imp} ext: line_rad")
    assert_matches(b["cont_rad"], r["cont_rad"], f"{imp} ext: cont_rad")
    # rhop_grid must actually be the supplied mapping, not the rho_vol fallback
    dev = np.max(np.abs(r["rhop"] - r["rvol"] / CASE["rvol_lcfs"]))
    assert dev > 1e-3, (
        "rhop_grid equals rvol/rvol_lcfs -- the external mapping was ignored")


@pytest.mark.parametrize("imp", ["C"])
def test_stepped_restart(imp):
    """The integrated-modelling coupling pattern (nz_init hand-off)."""
    r = _cached(imp + "_stepped", run_stepped, imp)
    b = load_baseline(imp + "_stepped")
    assert_matches(b["rvol"], r["rvol"], f"{imp} stepped: rvol_grid")
    assert_matches(b["nz_final"], r["nz_final"], f"{imp} stepped: nz final")


@_params("single")
def test_physical_sanity(result):
    """Invariants that hold regardless of the baseline.

    Assertions compare scalars so that a failure prints a number rather than a
    multi-megabyte array.
    """
    imp, r = result
    for key in ("nz", "line_rad", "cont_rad"):
        n_bad = int((~np.isfinite(r[key])).sum())
        assert n_bad == 0, f"{imp}: {n_bad} non-finite values in {key}"

    # line_rad is clamped inside compute_rad, so positivity IS an invariant
    worst = float(np.nanmin(r["line_rad"]))
    assert worst >= 0.0, f"{imp}: negative line_rad, min = {worst:.6e}"
    # cont_rad = nz[:, 1:] * prb inherits the nz undershoot, so it is not
    # required to be positive -- it is frozen instead, below.

    # a constant source must make the confined inventory grow monotonically
    tot = r["n_conf"].sum(1)
    worst = float(np.min(np.diff(tot)) / max(tot.max(), 1e-300))
    assert worst > -1e-6, (
        f"{imp}: confined inventory not monotonic under a constant source; "
        f"worst relative decrease {worst:.3e}")


@_params("single")
def test_negative_density_undershoot(result):
    """The scheme's positivity error, frozen per charge state.

    Not an invariant -- this solver undershoots by ~1-2 % of each charge
    state's peak in the near-zero regions. The point is that the refactor must
    not change it.
    """
    imp, r = result
    b = load_baseline(imp)
    assert_matches(b["undershoot"], r["undershoot"],
                   f"{imp}: nz undershoot per charge state", rtol=1e-6)
    ref_frac = float(b["neg_fraction"])
    now_frac = float(r["neg_fraction"])
    assert abs(now_frac - ref_frac) < 1e-6, (
        f"{imp}: fraction of negative nz entries changed "
        f"{ref_frac:.6f} -> {now_frac:.6f}")
    ref_c, now_c = float(b["cont_undershoot"]), float(r["cont_undershoot"])
    assert abs(now_c - ref_c) <= 1e-6 * max(abs(ref_c), 1e-30), (
        f"{imp}: continuum-radiation undershoot changed "
        f"{ref_c:.6e} -> {now_c:.6e}")


@pytest.mark.parametrize("imp", IMPURITIES)
@pytest.mark.parametrize("rm", ROTATION_MODELS)
def test_facit_coefficients(imp, rm):
    """Neoclassical Dz and Vconv per charge state.

    rotation_model=2 additionally guards the numpy-2 migration: it integrates
    with np.trapezoid, which replaced the np.trapz that numpy 2 removed.
    """
    name = f"{imp}_facit_rm{rm}"
    r = _cached(name, run_facit, imp, rm)
    b = load_baseline(name)
    _plot("facit", name, b, r, imp=imp)
    assert np.isfinite(r["Dz"]).all(), f"{name}: non-finite Dz"
    assert np.isfinite(r["Vconv"]).all(), f"{name}: non-finite Vconv"
    assert_matches(b["roa"], r["roa"], f"{name}: roa grid")
    assert_matches(b["Dz"], r["Dz"], f"{name}: FACIT Dz [m^2/s]")
    assert_matches(b["Vconv"], r["Vconv"], f"{name}: FACIT Vconv [m/s]")


def test_facit_multi_species(rotation_model=2):
    """C and W as two intrinsic impurities, coupled through a shared Zeff.

    Freezes the whole chain: two flat runs -> combined Zeff(r) -> FACIT per
    species with that Zeff -> unit conversion -> charge-state-resolved D/V ->
    two transport runs.
    """
    name = "multi_species"
    r = _cached(name, run_multi_species, rotation_model)
    b = load_baseline(name)
    _plot("multi", name, b, r)

    assert_matches(b["Zeff"], r["Zeff"], "multi: Zeff(r)")
    for imp in IMPURITIES:
        assert_matches(b[f"dZeff_{imp}"], r[f"dZeff_{imp}"],
                       f"multi: Zeff contribution from {imp}")
        for k, unit in (("Dz_si", "m^2/s"), ("Vconv_si", "m/s"),
                        ("D_z", "cm^2/s"), ("V_z", "cm/s")):
            assert_matches(b[f"{imp}_{k}"], r[f"{imp}_{k}"],
                           f"multi/{imp}: {k} [{unit}]")
        for k in ("nz", "line_rad", "cont_rad", "n_conf"):
            assert_matches(b[f"{imp}_{k}"], r[f"{imp}_{k}"], f"multi/{imp}: {k}")

    # both species must actually contribute to Zeff, else the coupling is idle
    for imp in IMPURITIES:
        assert np.nanmax(r[f"dZeff_{imp}"]) > 1e-3, \
            f"multi: {imp} contributes nothing to Zeff"
    assert np.nanmin(r["Zeff"]) >= 1.0, "multi: Zeff < 1 is unphysical"


def test_multi_species_differs_from_single(rotation_model=2):
    """The shared Zeff must change the answer versus a single-species run.

    Guards against the coupling silently doing nothing -- e.g. if Zeff were
    dropped, or a rotation model that ignores it were selected.
    """
    r = _cached("multi_species", run_multi_species, rotation_model)
    for imp in IMPURITIES:
        solo = _cached(f"{imp}_facit_rm{rotation_model}", run_facit,
                       imp, rotation_model)
        peak = np.nanmax(np.abs(solo["Dz"]))
        dev = np.nanmax(np.abs(r[f"{imp}_Dz_si"] - solo["Dz"])) / max(peak, 1e-300)
        assert dev > 1e-6, (
            f"{imp}: multi-species Dz is indistinguishable from the "
            f"single-species run (rel dev {dev:.2e}) -- Zeff coupling is idle")


# ---------------------------------------------------------------------------
# Baseline regeneration
# ---------------------------------------------------------------------------
def _regenerate():
    import aurora

    print(f"aurora {aurora.__version__} | numpy {np.__version__} | "
          f"python {platform.python_version()}")
    written = []
    for imp in IMPURITIES:
        print(f"  {imp}: single ...", end="", flush=True)
        written.append(save_baseline(imp, run_single(imp)))
        print(" ext ...", end="", flush=True)
        written.append(save_baseline(imp + "_ext",
                                     run_single(imp, external_grid=True)))
        print(" done")
    print("  C: stepped ...", end="", flush=True)
    written.append(save_baseline("C_stepped", run_stepped("C")))
    print(" done")
    for imp in IMPURITIES:
        for rm in ROTATION_MODELS:
            print(f"  {imp}: FACIT rm={rm} ...", end="", flush=True)
            written.append(save_baseline(f"{imp}_facit_rm{rm}",
                                         run_facit(imp, rm)))
            print(" done")
    print("  C+W: FACIT multi-species (shared Zeff) ...", end="", flush=True)
    written.append(save_baseline("multi_species", run_multi_species()))
    print(" done")
    print("\nwritten:")
    for p in written:
        print(f"  {os.path.relpath(p, HERE)}  "
              f"{os.path.getsize(p) / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        i = sys.argv.index("--plot")
        PLOT_DIR = (sys.argv[i + 1] if len(sys.argv) > i + 1
                    and not sys.argv[i + 1].startswith("-") else "outputs")
    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print(__doc__)
        sys.exit("pass --regenerate to write baselines")
