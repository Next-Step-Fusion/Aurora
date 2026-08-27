"""Shared helpers for the Aurora impurity-transport comparison tests.

Used by ``tests/test_with_omfit.py`` to run the same physics case twice --
once through the OMFIT geqdsk path and once through the OMFIT-free namelist
path -- and to compare the two.

Everything here is deliberately free of side effects at import time.
"""

import os

import numpy as np

# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------
# Two series only: the *method* carries the colour, the charge state is carried
# by panel position (small multiples). Both hues are from the Okabe-Ito
# colourblind-safe set and were checked against the chart palette rules:
#   lightness band ok | chroma floor ok | CVD dE 21.9 | normal-vision dE 31.2
#   | contrast vs white >= 3.0            -> all pass
C_MINE = "#0072B2"   # blue      -- namelist-only ("my") approach, solid
C_OMFIT = "#D55E00"  # vermillion-- OMFIT/geqdsk reference, dashed
C_GRID = "#c9c9c4"
C_INK = "#2b2b28"
C_MUTED = "#6f6f68"

# Solid is drawn thick and slightly transparent, dashed is drawn thin on top.
# When the two curves coincide *exactly* the dashes remain visible over the
# thick solid line, so "perfect agreement" is still legible.
STYLE_MINE = dict(color=C_MINE, lw=2.8, alpha=0.75, solid_capstyle="round", zorder=2)
STYLE_OMFIT = dict(color=C_OMFIT, lw=1.5, ls=(0, (5, 3)), zorder=3)

LBL_MINE = "external grid (no OMFIT)"
LBL_OMFIT = "OMFIT geqdsk"


def apply_style():
    """Recessive grid/axes, text in ink tokens rather than series colours."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": C_MUTED,
        "axes.labelcolor": C_INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.color": C_INK,
        "font.size": 9,
    })


def outdir(path="tests/output"):
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Input data
# --------------------------------------------------------------------------
def load_state(h5_path="tests/data/centaur.h5", idt=-1):
    """Read the equilibrium + kinetic profiles from the Centaur state file.

    Returns a plain dict so that neither test needs to know the HDF5 layout.

    Units in the file (verified against the stored values):
      * ``electron_density_profile``    [1e19 m^-3]   -> x1e13 gives cm^-3
      * ``electron_temperature_profile`` [eV]         -> used as-is
      * ``ion_temperature_profile``      [eV]         -> used as-is
    """
    import h5py

    h5 = h5py.File(h5_path, "r")
    st = h5["history/state"]

    rho_tor = st["normalized_rho"][idt]

    # rho_pol = sqrt(normalized poloidal flux). Aurora's namelist "rhop" field
    # is rho_POLOIDAL (docs/params.rst), so this -- not rho_tor -- is what the
    # kinetic profiles must be given on.
    psi_profile = st["psi_profile"][idt]
    psi_N = (psi_profile - psi_profile[0]) / (psi_profile[-1] - psi_profile[0])
    rho_pol = np.sqrt(np.clip(psi_N, 0.0, None))

    eq = st["eqdsk_plasma_equilibrium_data"][idt]
    out = dict(
        rho_tor=rho_tor,
        rho_pol=rho_pol,
        volume=st["volume"][idt],                    # [m^3]
        q=st["q"][idt],
        aminor=st["minor_radius"][idt],              # [m]
        ne=st["electron_density_profile"][idt],      # [1e19 m^-3]
        Te=st["electron_temperature_profile"][idt],  # [eV]
        Ti=st["ion_temperature_profile"][idt],       # [eV]
        psi_RZ=st["psi"][idt],
        R=st["x_psi"][idt],
        Z=st["y_psi"][idt],
        eq=eq,
        rcentr=eq[2],
        rmagx=eq[5],
        bcentr=eq[9],
        boundary=st["boundary"][idt],
        fpol=st["eqdsk_data_fpol_out"][idt],
        pres=st["eqdsk_data_pres_out"][idt],
        ffprime=st["eqdsk_data_ffprime_out"][idt],
        pprime=st["eqdsk_data_pprime_out"][idt],
        qpsi=st["eqdsk_data_qpsi_out"][idt],
    )
    # background neutral density [1e19 m^-3], same parameterisation as the
    # original scripts
    out["n0"] = 0.001 * np.exp((rho_tor - 1.0) / 0.02)
    return out


_RLIM = np.array([1.26, 1.44999999999999, 2.17799999999999, 2.17799999999999, 2.28299999999999,
                  2.46303142, 2.7, 2.67399999999999, 2.6, 2.72999999999999, 2.72999999999999,
                  2.6, 2.67399999999999, 2.7, 2.46303142, 2.28299999999999, 2.17799999999999,
                  2.17799999999999, 1.44999999999999, 1.26, 1.26])
_ZLIM = np.array([-0.27, -0.689999999999999, -1.19999999999999, -1.29299999999999, -1.56,
                  -1.37471264, -1.41999999999999, -1.23749999999999, -1.1, -0.45517241,
                  0.45517241, 1.1, 1.23749999999999, 1.41999999999999, 1.37471264, 1.56,
                  1.29299999999999, 1.19999999999999, 0.689999999999999, 0.27, -0.27])


def write_geqdsk(state, path="tests/data/centaur.geqdsk"):
    """Write the g-EQDSK that the OMFIT branch reads back in."""
    from freeqdsk import geqdsk as _geqdsk

    eq, bnd = state["eq"], state["boundary"]
    nb = len(bnd) - 1
    data = {
        "comment": "999",
        "shot": 0,
        "nx": len(state["R"]),
        "ny": len(state["Z"]),
        "rdim": eq[0], "zdim": eq[1], "rcentr": eq[2], "rleft": eq[3], "zmid": eq[4],
        "rmagx": eq[5], "zmagx": eq[6], "simagx": eq[7], "sibdry": eq[8],
        "bcentr": eq[9], "cpasma": eq[10],
        "fpol": state["fpol"], "pres": state["pres"],
        "ffprime": state["ffprime"], "pprime": state["pprime"],
        "psi": state["psi_RZ"] / (2 * np.pi),   # [Wb/rad]
        "qpsi": state["qpsi"],
        "nbdry": nb, "rbdry": bnd[:, 0][:nb], "zbdry": bnd[:, 1][:nb],
        "nlim": len(_RLIM), "rlim": _RLIM, "zlim": _ZLIM,
    }
    with open(path, "w") as fid:
        _geqdsk.write(data, fid)
    return path


def rvol_lcfs_from_volume(volume, R0_m):
    """rvol_lcfs [cm] = 100 * sqrt(V_LCFS / (2 pi^2 R0)).

    ``R0_m`` must be the *magnetic axis* major radius (RMAXIS), because that is
    what Aurora uses internally in ``get_rhopol_rvol_mapping`` and ``vol_int``.
    """
    return float(np.sqrt(volume[-1] / (2 * np.pi**2 * R0_m)) * 100.0)


# ---------------------------------------------------------------------------
# WORKAROUND (remove when the state-file producer is patched)
# ---------------------------------------------------------------------------
# The `volume` array in centaur.h5 is stored against the `normalized_rho`
# (rho_tor) index, but its values are the volumes of surfaces labelled by
# rho_POL. Because both coordinates are linspace(0, 1, 24) the mixup is
# invisible from the array shapes alone.
#
# Evidence: inverting V_state through the equilibrium's true V(rho_pol) gives
# an implied rho_pol that tracks rho_tor to ~0.005, not rho_pol. Pairing V with
# rho_tor reduces the rho_pol(rvol) error against the geqdsk mapping from
# 0.087 to 0.008 -- the latter being the intrinsic state-vs-geqdsk scatter.
#
# Set VOLUME_COORD = "rho_pol" once the producer writes V on the rho_tor grid.
VOLUME_COORD = "rho_tor"


def external_grid_from_state(state, bound_sep=2.0, K=6.0, dr_0=0.3, dr_1=0.05,
                             volume_coord=None):
    """Build (rvol_grid, rhop_grid, rvol_lcfs) from the equilibrium alone.

    This is what an integrated-modelling driver would hand Aurora instead of a
    geqdsk. The equilibrium supplies the two things Aurora actually needs:

      1. ``rvol_lcfs`` -- from the enclosed volume at the LCFS
      2. the ``rvol <-> rho_pol`` mapping -- from V(rho_pol)

    The *spacing* of the interior points is a free discretisation choice; here
    Aurora's own STRAHL law is used so that resolution matches a geqdsk-based
    run and the comparison is like-for-like.

    Outside the LCFS the equilibrium carries no rho_pol information, so the
    mapping is extrapolated with the local d(rho_pol)/d(rvol) at the separatrix.
    That is first-order correct, unlike the rho_vol fallback (which assumes
    d(rho_pol)/d(rvol) = 1/rvol_lcfs everywhere).

    Returns
    -------
    rvol_grid : 1D array [cm]
    rhop_grid : 1D array, rho_pol at each rvol_grid point
    rvol_lcfs : float [cm]
    """
    import aurora

    R0_m = state["rmagx"]
    # coordinate the `volume` array is really tabulated against -- see the
    # WORKAROUND note above
    volume_coord = volume_coord or VOLUME_COORD
    rho_pol = np.asarray(state[volume_coord], dtype=float)
    V = np.asarray(state["volume"], dtype=float)

    # the equilibrium's own rvol(rho_pol)
    rvol_eq = np.sqrt(V / (2 * np.pi**2 * R0_m)) * 100.0     # m -> cm
    rvol_eq[0] = 0.0
    rvol_lcfs = float(rvol_eq[-1])

    nml = dict(K=K, dr_0=dr_0, dr_1=dr_1,
               rvol_lcfs=rvol_lcfs, bound_sep=bound_sep, lim_sep=1.0)
    rvol_grid = aurora.grids_utils.create_radial_grid(nml)[0]

    rhop_grid = np.interp(rvol_grid, rvol_eq, rho_pol)
    # SOL: linear extrapolation using the true edge gradient
    slope = (rho_pol[-1] - rho_pol[-2]) / (rvol_eq[-1] - rvol_eq[-2])
    sol = rvol_grid > rvol_lcfs
    rhop_grid[sol] = 1.0 + slope * (rvol_grid[sol] - rvol_lcfs)
    rhop_grid[0] = 0.0
    return rvol_grid, rhop_grid, rvol_lcfs


# --------------------------------------------------------------------------
# Running Aurora
# --------------------------------------------------------------------------
def make_namelist(state, rhop_prof, imp="C", geqdsk=None,
                  rvol_lcfs_cm=None, source_rate=1e21,
                  rvol_grid=None, rhop_grid=None):
    """Build the Aurora namelist shared by both branches.

    ``rhop_prof`` is the radial coordinate the kinetic profiles are given on and
    must be rho_POLOIDAL. Everything else (species, source, transport) is
    identical between the two branches so that the only difference under test is
    how the radial grid and the rho_pol mapping are obtained.
    """
    import aurora

    nml = aurora.load_default_namelist()

    if geqdsk is None:
        # OMFIT-free branch: the equilibrium scalars must be supplied by hand.
        nml["Baxis"] = state["bcentr"]
        nml["Raxis_cm"] = state["rmagx"] * 1e2
        nml["rvol_lcfs"] = rvol_lcfs_cm
        if rvol_grid is not None:
            nml["rvol_grid"] = rvol_grid       # external radial grid
        if rhop_grid is not None:
            nml["rhop_grid"] = rhop_grid       # external rho_pol mapping

    nml["kin_profs"]["ne"]["rhop"] = rhop_prof
    nml["kin_profs"]["ne"]["vals"] = state["ne"] * 1e13   # 1e19 m^-3 -> cm^-3
    nml["kin_profs"]["Te"]["rhop"] = rhop_prof
    nml["kin_profs"]["Te"]["vals"] = state["Te"]          # already eV
    nml["kin_profs"]["Ti"]["rhop"] = rhop_prof
    nml["kin_profs"]["Ti"]["vals"] = state["Ti"]          # already eV
    nml["kin_profs"]["n0"]["rhop"] = rhop_prof
    nml["kin_profs"]["n0"]["vals"] = state["n0"] * 1e13   # 1e19 m^-3 -> cm^-3

    nml["imp"] = imp
    nml["source_type"] = "const"
    nml["source_rate"] = source_rate
    nml["LBO"] = None
    return nml


def run_case(state, rhop_prof, time_grid, geqdsk=None, rvol_lcfs_cm=None,
             imp="C", D_cm2s=1e4, V_cms=0.0, label="", verbose=True,
             rvol_grid=None, rhop_grid=None):
    """Advance Aurora over ``time_grid``, feeding each step's result forward.

    Mirrors the structure of the original scripts (rebuild the sim each step,
    pass the previous ``nz`` in as ``nz_init``).

    Returns a dict with the full nz history plus the grids needed to compare.
    """
    import aurora

    dt = time_grid[1] - time_grid[0]
    nml = make_namelist(state, rhop_prof, imp=imp, geqdsk=geqdsk,
                        rvol_lcfs_cm=rvol_lcfs_cm,
                        rvol_grid=rvol_grid, rhop_grid=rhop_grid)

    nz = None
    nz_time = []
    asim = None
    for it, t in enumerate(time_grid):
        nml["timing"] = {
            "dt_increase": np.array([1.0, 1.0]),
            "dt_start": np.array([dt, dt]),
            "steps_per_cycle": np.array([1, 1]),
            "times": np.array([t, t + dt]),
        }
        asim = aurora.aurora_sim(nml, geqdsk=geqdsk) if geqdsk is not None \
            else aurora.aurora_sim(nml)

        if nz is None:                       # size the initial state from the grid
            nz = np.zeros((len(asim.rvol_grid), asim.Z_imp + 1))

        D_z = D_cm2s * np.ones(len(asim.rvol_grid))
        V_z = V_cms * np.ones(len(asim.rvol_grid))
        out = asim.run_aurora(D_z, V_z, nz_init=nz)
        nz = np.maximum(out["nz"][:, :, -1], 0.0)
        nz_time.append(nz.copy())

        if verbose and (it % 20 == 0 or it == len(time_grid) - 1):
            print(f"    [{label}] step {it + 1:3d}/{len(time_grid)}  "
                  f"nz(0,C3+) = {nz[0, 3]:.4e} cm^-3")

    nz_time = np.asarray(nz_time)            # (nt, nr, nZ)
    return dict(
        label=label,
        nz_time=nz_time,
        nz_final=nz_time[-1],
        rhop_grid=asim.rhop_grid,
        rvol_grid=asim.rvol_grid,
        pro_grid=asim.pro_grid,
        rvol_lcfs=asim.rvol_lcfs,
        Raxis_cm=asim.Raxis_cm,
        Z_imp=asim.Z_imp,
        imp=imp,
        ne=asim._ne[-1] if hasattr(asim, "_ne") else None,
        Te=asim._Te[-1] if hasattr(asim, "_Te") else None,
        asim=asim,
    )


def compute_radiation(case):
    """Line and continuum radiation for the final state, in W/cm^3.

    ``line_rad`` comes from ADAS "plt" files, ``cont_rad`` from "prb"
    (recombination + bremsstrahlung continuum).
    """
    import aurora

    nz = case["nz_final"].T[None]                     # (1, nZ, nr)
    ne = np.atleast_2d(case["ne"])
    Te = np.atleast_2d(case["Te"])
    res = aurora.radiation.compute_rad(
        case["imp"], nz, ne, Te, prad_flag=True,
        thermal_cx_rad_flag=False, spectral_brem_flag=False, sxr_flag=False,
    )
    return dict(
        line_rad=res["line_rad"][0],     # (nZ, nr)
        cont_rad=res["cont_rad"][0],     # (nZ, nr)
        tot=res["tot"][0],               # (nr,)
    )


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def to_common_grid(case, y, rhop_common, axis=-1):
    """Interpolate ``y`` from a case's own rhop grid onto a shared grid."""
    return np.interp(rhop_common, case["rhop_grid"], y)


def volume_integral(case, prof):
    """Volume integral over the confined region, using each case's own grid."""
    import aurora

    return float(aurora.grids_utils.vol_int(
        prof, case["rvol_grid"], case["pro_grid"], case["Raxis_cm"],
        rvol_max=case["rvol_lcfs"]))


def profile_stats(ref, tst, rhop_common, ref_case, tst_case):
    """Difference metrics between two profiles already on ``rhop_common``."""
    scale = np.nanmax(np.abs(ref))
    if scale == 0:
        scale = 1.0
    diff = tst - ref
    rel = 100.0 * diff / scale
    return dict(
        ref_peak=float(np.nanmax(ref)),
        tst_peak=float(np.nanmax(tst)),
        peak_pct=float(100.0 * (np.nanmax(tst) - np.nanmax(ref)) / scale),
        max_abs_pct=float(np.nanmax(np.abs(rel))),
        rms_pct=float(np.sqrt(np.nanmean(rel**2))),
        rhop_at_ref_peak=float(rhop_common[np.nanargmax(ref)]),
        rhop_at_tst_peak=float(rhop_common[np.nanargmax(tst)]),
    )


def auto_xlim(curves, frac=0.005, pad=0.12, min_width=0.18, hard_max=1.05):
    """x-range that actually contains the signal.

    Low charge states of a light impurity live entirely in a thin edge layer;
    plotting them on 0..1 wastes ~95% of the panel. ``curves`` is a list of
    ``(x, y)`` pairs -- the returned window covers every point where any curve
    exceeds ``frac`` of its own peak, padded and clamped.
    """
    lo, hi = np.inf, -np.inf
    for x, y in curves:
        y = np.asarray(y)
        peak = np.nanmax(np.abs(y))
        if peak <= 0:
            continue
        m = np.abs(y) > frac * peak
        if not m.any():
            continue
        lo = min(lo, float(np.asarray(x)[m].min()))
        hi = max(hi, float(np.asarray(x)[m].max()))
    if not np.isfinite(lo):
        return 0.0, hard_max
    w = hi - lo
    lo, hi = lo - pad * w, hi + pad * w
    if hi - lo < min_width:                      # keep very narrow peaks legible
        c = 0.5 * (lo + hi)
        lo, hi = c - 0.5 * min_width, c + 0.5 * min_width
    return max(0.0, lo), min(hard_max, hi)


def fmt_table(rows, headers, widths=None):
    """Small fixed-width table formatter (no pandas dependency)."""
    widths = widths or [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2
                        for i, h in enumerate(headers)]
    line = "".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "-" * len(line.rstrip())]
    for r in rows:
        out.append("".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)
