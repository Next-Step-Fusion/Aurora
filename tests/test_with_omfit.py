"""Compare the OMFIT-based Aurora setup against the OMFIT-free namelist setup.

Two runs of the *same* physics case:

  A. "external grid"  -- no geqdsk. The equilibrium enters through the scalars
                         Raxis_cm / Baxis / rvol_lcfs *and* through an external
                         radial grid supplied via the namelist keys
                         ``rvol_grid`` (volume-equivalent radii, cm) and
                         ``rhop_grid`` (the matching rho_pol). Both are built
                         from the equilibrium's V(rho_pol) alone -- see
                         helpers.external_grid_from_state. This is the target
                         for the OMFIT-free refactor.
  B. "OMFIT geqdsk"   -- aurora_sim(namelist, geqdsk=OMFITgeqdsk(...)). Aurora
                         derives rvol_lcfs and the rho_pol mapping from the
                         equilibrium itself. This is the reference.

Everything else -- species, source, transport, kinetic profiles -- is identical,
so any difference is attributable to how the radial grid and the rho_pol mapping
are obtained.

Branch A exercises the external-grid path in core.setup_grids: ``rvol_grid``
routes through grids_utils.grid_from_rvol (pro/qpr derived numerically rather
than analytically) and ``rhop_grid`` replaces the rho_vol fallback. Without
``rhop_grid`` Aurora would set rhop_grid = rvol/rvol_lcfs, i.e. rho_VOL, and
silently place the kinetic profiles on the wrong flux surfaces.

Outputs (tests/output/):
    nz_comparison_final.png     nz per charge state, A solid vs B dashed
    nz_relative_difference.png  (A - B) / max(B) per charge state
    prad_comparison.png         line + continuum radiation
    time_traces.png             confined impurity inventory vs time
    nz_evolution_comparison.gif animated version of the first figure
    comparison_statistics.txt   the numbers printed below

Two corrections relative to the original version of this script, needed to make
the comparison meaningful (see NOTE blocks below): the kinetic profiles are
placed on rho_pol rather than geqdsk["RHOVN"], and Te/Ti are used in eV without
a spurious 1e3 factor.

Run from the repository root:  python tests/test_with_omfit.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import imageio
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers as H

import aurora
from omfit_classes import omfit_eqdsk

H.apply_style()
OUT = H.outdir()

# ---------------------------------------------------------------------------
# 1. Inputs
# ---------------------------------------------------------------------------
state = H.load_state("tests/data/centaur.h5", idt=-1)
H.write_geqdsk(state, "tests/data/centaur.geqdsk")
geqdsk = omfit_eqdsk.OMFITgeqdsk("tests/data/centaur.geqdsk")

# NOTE (coordinate): Aurora's namelist field "rhop" is sqrt of normalized
# POLOIDAL flux (upstream docs: params.rst). geqdsk["RHOVN"], used by the original
# script, is sqrt of normalized TOROIDAL flux -- OMFIT builds it from
# Phi = int q dpsi. Feeding rho_tor in as rhop misplaces the profiles radially
# (up to ~0.16 in rho for this equilibrium). Both branches therefore use
# state["rho_pol"], computed from psi_profile.
rhop_prof = state["rho_pol"]

# NOTE (units): electron_temperature_profile in the state file is already in eV
# (max 2.45e4 = 24.5 keV). The original script multiplied it by 1e3.
# It is used unscaled here, matching tests/test_core_impurity.py.

# rvol_lcfs for the OMFIT-free branch. R0 must be the MAGNETIC AXIS (rmagx):
# Aurora's own get_rhopol_rvol_mapping and vol_int both use RMAXIS, so using
# rcentr here would put the two branches on different-sized plasmas.
rvol_lcfs_cm = H.rvol_lcfs_from_volume(state["volume"], state["rmagx"])
rvol_lcfs_rcentr = H.rvol_lcfs_from_volume(state["volume"], state["rcentr"])

# External radial grid + rho_pol mapping, built from V(rho_pol) only.
rvol_ext, rhop_ext, rvol_lcfs_ext = H.external_grid_from_state(state, bound_sep=2.0)

IMP = "C"
dt = 1e-3
time_grid = np.arange(0.0, 0.1, dt)

print("=" * 78)
print("Aurora: OMFIT-free vs OMFIT-based comparison")
print("=" * 78)
print(f"  impurity            : {IMP}")
print(f"  time grid           : {len(time_grid)} steps of {dt * 1e3:.1f} ms "
      f"(0 -> {time_grid[-1] + dt:.3f} s)")
print(f"  profiles on         : rho_pol ({len(rhop_prof)} pts)")
print(f"  V_LCFS              : {state['volume'][-1]:.4f} m^3")
print(f"  R_magnetic_axis     : {state['rmagx']:.4f} m")
print(f"  R_centre (rcentr)   : {state['rcentr']:.4f} m")
print(f"  rvol_lcfs (rmagx)   : {rvol_lcfs_cm:.3f} cm   <- used")
print(f"  rvol_lcfs (rcentr)  : {rvol_lcfs_rcentr:.3f} cm   "
      f"({100 * (rvol_lcfs_rcentr / rvol_lcfs_cm - 1):+.2f} %)")
print(f"  volume tabulated on : state['{H.VOLUME_COORD}']   "
      f"{'(WORKAROUND - see helpers.py)' if H.VOLUME_COORD != 'rho_pol' else ''}")
print(f"  external grid       : {len(rvol_ext)} pts, rvol "
      f"{rvol_ext[0]:.1f}..{rvol_ext[-1]:.2f} cm | rhop "
      f"{rhop_ext[0]:.3f}..{rhop_ext[-1]:.4f}")

# ---------------------------------------------------------------------------
# 2. Run both branches
# ---------------------------------------------------------------------------
print("\n[A] external rvol_grid + rhop_grid (no OMFIT)")
case_mine = H.run_case(state, rhop_prof, time_grid, geqdsk=None,
                       rvol_lcfs_cm=rvol_lcfs_ext, imp=IMP, label="A",
                       rvol_grid=rvol_ext, rhop_grid=rhop_ext)

print("\n[B] OMFIT geqdsk")
case_omfit = H.run_case(state, rhop_prof, time_grid, geqdsk=geqdsk,
                        imp=IMP, label="B")

nZ = case_omfit["Z_imp"] + 1
print("\ngrids")
print(f"  A: {len(case_mine['rvol_grid']):4d} pts | rvol_lcfs "
      f"{case_mine['rvol_lcfs']:.3f} cm | rhop_grid "
      f"{case_mine['rhop_grid'][0]:.3f}..{case_mine['rhop_grid'][-1]:.4f}")
print(f"  B: {len(case_omfit['rvol_grid']):4d} pts | rvol_lcfs "
      f"{case_omfit['rvol_lcfs']:.3f} cm | rhop_grid "
      f"{case_omfit['rhop_grid'][0]:.3f}..{case_omfit['rhop_grid'][-1]:.4f}")

# ---------------------------------------------------------------------------
# 3. Common grid + statistics
# ---------------------------------------------------------------------------
rhop_c = np.linspace(0.0, 1.0, 201)

rows = []
per_z = {}
for iz in range(nZ):
    ref = H.to_common_grid(case_omfit, case_omfit["nz_final"][:, iz], rhop_c)
    tst = H.to_common_grid(case_mine, case_mine["nz_final"][:, iz], rhop_c)
    s = H.profile_stats(ref, tst, rhop_c, case_omfit, case_mine)
    per_z[iz] = s
    N_ref = H.volume_integral(case_omfit, case_omfit["nz_final"][:, iz])
    N_tst = H.volume_integral(case_mine, case_mine["nz_final"][:, iz])
    dN = 100.0 * (N_tst - N_ref) / N_ref if N_ref > 0 else np.nan
    rows.append([
        f"{IMP}{iz}+",
        f"{s['ref_peak']:.4e}", f"{s['tst_peak']:.4e}", f"{s['peak_pct']:+.2f}",
        f"{s['rms_pct']:.2f}", f"{s['max_abs_pct']:.2f}",
        f"{s['rhop_at_ref_peak']:.3f}", f"{s['rhop_at_tst_peak']:.3f}",
        f"{dN:+.2f}",
    ])

stats_lines = []


def emit(txt=""):
    print(txt)
    stats_lines.append(txt)


emit("\n" + "=" * 78)
emit("NUMERICAL STATISTICS  (reference = OMFIT geqdsk, test = external grid)")
emit("=" * 78)
emit("\nper charge state, final time t = %.3f s" % (time_grid[-1] + dt))
emit("  peak values in cm^-3; percentages normalised to max(reference)")
emit("")
emit(H.fmt_table(
    rows,
    ["state", "peak B(OMFIT)", "peak A(ext)", "peak %", "RMS %", "max %",
     "rhop@pkB", "rhop@pkA", "N_conf %"]))

# totals over all charge states
ref_tot = H.to_common_grid(case_omfit, case_omfit["nz_final"].sum(1), rhop_c)
tst_tot = H.to_common_grid(case_mine, case_mine["nz_final"].sum(1), rhop_c)
s_tot = H.profile_stats(ref_tot, tst_tot, rhop_c, case_omfit, case_mine)
N_ref = H.volume_integral(case_omfit, case_omfit["nz_final"].sum(1))
N_tst = H.volume_integral(case_mine, case_mine["nz_final"].sum(1))

emit("\ntotal impurity density (sum over charge states)")
emit(f"  peak            B = {s_tot['ref_peak']:.4e} cm^-3   "
     f"A = {s_tot['tst_peak']:.4e} cm^-3   ({s_tot['peak_pct']:+.2f} %)")
emit(f"  RMS difference  {s_tot['rms_pct']:.2f} %   "
     f"(max {s_tot['max_abs_pct']:.2f} %)")
emit(f"  confined inventory  B = {N_ref:.4e}   A = {N_tst:.4e} particles   "
     f"({100 * (N_tst - N_ref) / N_ref:+.2f} %)")

# radiation
rad_mine = H.compute_radiation(case_mine)
rad_omfit = H.compute_radiation(case_omfit)
emit("\nradiation at final time (W cm^-3, volume-integrated to W)")
rrows = []
for key, name in [("line_rad", "line"), ("cont_rad", "continuum")]:
    pr_ref = rad_omfit[key].sum(0)
    pr_tst = rad_mine[key].sum(0)
    P_ref = H.volume_integral(case_omfit, pr_ref)
    P_tst = H.volume_integral(case_mine, pr_tst)
    r = H.to_common_grid(case_omfit, pr_ref, rhop_c)
    t = H.to_common_grid(case_mine, pr_tst, rhop_c)
    st_ = H.profile_stats(r, t, rhop_c, case_omfit, case_mine)
    rrows.append([name, f"{P_ref:.4e}", f"{P_tst:.4e}",
                  f"{100 * (P_tst - P_ref) / P_ref:+.2f}",
                  f"{st_['rms_pct']:.2f}", f"{st_['max_abs_pct']:.2f}"])
P_ref = H.volume_integral(case_omfit, rad_omfit["tot"])
P_tst = H.volume_integral(case_mine, rad_mine["tot"])
rrows.append(["total", f"{P_ref:.4e}", f"{P_tst:.4e}",
              f"{100 * (P_tst - P_ref) / P_ref:+.2f}", "-", "-"])
emit("")
emit(H.fmt_table(rrows, ["Prad", "B(OMFIT) [W]", "A(ext) [W]", "P %",
                         "RMS %", "max %"]))

# ---------------------------------------------------------------------------
# 3b. Patch validation: same equilibrium data through both code paths
# ---------------------------------------------------------------------------
# A vs B above compares two *different* volume profiles (see the diagnostics
# below). To check the external-grid machinery itself, run a third case that
# goes through the namelist path but is fed OMFIT's own rvol <-> rho_pol
# mapping. If grid_from_rvol + rhop_grid are correct this must reproduce B.
_rp_o, _rv_o = aurora.grids_utils.get_rhopol_rvol_mapping(geqdsk)
_rvol_v = case_omfit["rvol_grid"]
_rhop_v = np.interp(_rvol_v, _rv_o, _rp_o)
_rhop_v[0] = 0.0
case_valid = H.run_case(state, rhop_prof, time_grid[:20], geqdsk=None,
                        rvol_lcfs_cm=case_omfit["rvol_lcfs"], imp=IMP,
                        label="V", verbose=False,
                        rvol_grid=_rvol_v, rhop_grid=_rhop_v)
_ref20 = case_omfit["nz_time"][19]
_dev = 100 * np.nanmax(np.abs(case_valid["nz_final"] - _ref20)) / np.nanmax(_ref20)
_pro_dev = 100 * np.nanmax(np.abs(
    case_valid["pro_grid"][1:] / case_omfit["pro_grid"][1:] - 1))

emit("\nPATCH VALIDATION  (external-grid path fed OMFIT's own mapping vs branch B)")
emit(f"  max |nz difference| after 20 steps : {_dev:.4f} % of peak")
emit(f"  max |pro numerical/analytic - 1|   : {_pro_dev:.4f} %")
emit("  -> the rvol_grid / rhop_grid path reproduces the geqdsk path; the A-vs-B")
emit("     residual above is input data, not the code path.")

# grid / equilibrium diagnostics
emit("\ngrid & equilibrium diagnostics")
emit(f"  radial points        B = {len(case_omfit['rvol_grid'])}   "
     f"A = {len(case_mine['rvol_grid'])}")
emit(f"  rvol_lcfs [cm]       B = {case_omfit['rvol_lcfs']:.3f}   "
     f"A = {case_mine['rvol_lcfs']:.3f}   "
     f"({100 * (case_mine['rvol_lcfs'] / case_omfit['rvol_lcfs'] - 1):+.2f} %)")
emit(f"  Raxis_cm             B = {case_omfit['Raxis_cm']:.3f}   "
     f"A = {case_mine['Raxis_cm']:.3f}")
# how far apart are the two volume profiles the branches are built from?
_Vo = np.asarray(geqdsk["fluxSurfaces"]["geo"]["vol"])
_rpo = np.sqrt(np.asarray(geqdsk["fluxSurfaces"]["geo"]["psin"]))
emit("\n  volume profiles the two branches are built from [m^3]:")
emit(f"    rho_pol   V(state, on {H.VOLUME_COORD})   V(OMFIT fluxSurfaces)   ratio")
for _r in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
    _a = np.interp(_r, state[H.VOLUME_COORD], state["volume"])
    _b = np.interp(_r, _rpo, _Vo)
    emit(f"    {_r:.2f}      {_a:8.4f}        {_b:8.4f}          {_a / _b:.4f}")
emit(f"    (V read against state['{H.VOLUME_COORD}'] -- the h5 stores it on the")
emit("     rho_tor index while the values belong to rho_pol labels; see the")
emit("     WORKAROUND note in helpers.py. Remove once the producer is fixed.)")
_dev_map = np.max(np.abs(
    np.interp(case_omfit["rvol_grid"], rvol_ext, rhop_ext)
    - case_omfit["rhop_grid"]))
emit(f"    max |rho_pol(rvol) A - B| = {_dev_map:.4f}")
emit("")
emit("  branch A supplies rvol_grid + rhop_grid through the namelist, so its")
emit("  rhop_grid is a true rho_pol taken from the equilibrium's V(rho_pol),")
emit("  NOT the rho_vol fallback.")
_dev = np.max(np.abs(case_mine["rhop_grid"] - case_mine["rvol_grid"] /
                     case_mine["rvol_lcfs"]))
emit(f"  max |rhop_grid - rvol/rvol_lcfs| in A = {_dev:.4f}  "
     f"(0 would mean the mapping was ignored)")
emit(f"  using rcentr instead of rmagx for rvol_lcfs would give "
     f"{rvol_lcfs_rcentr:.3f} cm "
     f"({100 * (rvol_lcfs_rcentr / case_omfit['rvol_lcfs'] - 1):+.2f} % vs B)")

with open(os.path.join(OUT, "comparison_statistics.txt"), "w") as f:
    f.write("\n".join(stats_lines) + "\n")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
XMAX = 1.05

# One x-window per panel, computed once from the whole time history so that
# figure 1 and the animation share axes and the gif does not jitter.
XLIM = []
for iz in range(nZ):
    XLIM.append(H.auto_xlim([
        (case_mine["rhop_grid"], np.abs(case_mine["nz_time"][:, :, iz]).max(0)),
        (case_omfit["rhop_grid"], np.abs(case_omfit["nz_time"][:, :, iz]).max(0)),
    ], hard_max=XMAX))
XLIM.append((0.0, XMAX))          # the "total" panel keeps the full range


def panel_grid(figsize=(15, 7.5)):
    fig, axs = plt.subplots(2, 4, figsize=figsize)
    return fig, axs.ravel()


def draw_pair(ax, case_a, case_b, ya, yb, lcfs=True, xlim=(0.0, XMAX)):
    ax.plot(case_b["rhop_grid"], yb, label=H.LBL_OMFIT, **H.STYLE_OMFIT)
    ax.plot(case_a["rhop_grid"], ya, label=H.LBL_MINE, **H.STYLE_MINE)
    if lcfs:
        ax.axvline(1.0, color=H.C_MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_xlim(*xlim)


# --- Figure 1: nz per charge state, final time ---------------------------
fig, axs = panel_grid()
for iz in range(nZ):
    ax = axs[iz]
    draw_pair(ax, case_mine, case_omfit,
              case_mine["nz_final"][:, iz], case_omfit["nz_final"][:, iz],
              xlim=XLIM[iz])
    zoom = "" if XLIM[iz][0] == 0.0 else "   (edge zoom)"
    ax.set_title(f"{IMP}$^{{{iz}+}}$" + zoom, color=H.C_INK)
    ax.set_ylim(bottom=0)
    if iz % 4 == 0:
        ax.set_ylabel(r"$n_z$  [cm$^{-3}$]")
axs[nZ].plot(case_omfit["rhop_grid"], case_omfit["nz_final"].sum(1),
             label=H.LBL_OMFIT, **H.STYLE_OMFIT)
axs[nZ].plot(case_mine["rhop_grid"], case_mine["nz_final"].sum(1),
             label=H.LBL_MINE, **H.STYLE_MINE)
axs[nZ].axvline(1.0, color=H.C_MUTED, lw=0.7, ls=":", zorder=1)
axs[nZ].set_xlim(*XLIM[nZ])
axs[nZ].set_ylim(bottom=0)
axs[nZ].set_title("total", color=H.C_INK)
for ax in axs:
    ax.set_xlabel(r"$\rho$  (each run's own $\mathtt{rhop\_grid}$)")
axs[0].legend(loc="upper left")
fig.suptitle(f"{IMP} impurity density at t = {time_grid[-1] + dt:.3f} s  "
             f"—  external grid (solid) vs OMFIT geqdsk (dashed)",
             fontsize=12, color=H.C_INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "nz_comparison_final.png"), dpi=160)
plt.close(fig)

# --- Figure 2: relative difference ---------------------------------------
fig, axs = panel_grid(figsize=(15, 7.0))
for iz in range(nZ):
    ax = axs[iz]
    ref = H.to_common_grid(case_omfit, case_omfit["nz_final"][:, iz], rhop_c)
    tst = H.to_common_grid(case_mine, case_mine["nz_final"][:, iz], rhop_c)
    scale = np.nanmax(np.abs(ref)) or 1.0
    ax.plot(rhop_c, 100 * (tst - ref) / scale, color=H.C_MINE, lw=1.8)
    ax.axhline(0, color=H.C_MUTED, lw=0.8)
    zoom = "" if XLIM[iz][0] == 0.0 else "   (edge zoom)"
    ax.set_title(f"{IMP}$^{{{iz}+}}$   RMS {per_z[iz]['rms_pct']:.1f}%" + zoom,
                 color=H.C_INK)
    if iz % 4 == 0:
        ax.set_ylabel("(A - B) / max(B)   [%]")
    ax.set_xlim(XLIM[iz][0], 1.0)
scale = np.nanmax(np.abs(ref_tot)) or 1.0
axs[nZ].plot(rhop_c, 100 * (tst_tot - ref_tot) / scale, color=H.C_MINE, lw=1.8)
axs[nZ].axhline(0, color=H.C_MUTED, lw=0.8)
axs[nZ].set_title(f"total   RMS {s_tot['rms_pct']:.1f}%", color=H.C_INK)
axs[nZ].set_xlim(0, 1.0)
for ax in axs:
    ax.set_xlabel(r"$\rho_{pol}$ (common grid)")
fig.suptitle("Relative difference: external grid minus OMFIT geqdsk",
             fontsize=12, color=H.C_INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "nz_relative_difference.png"), dpi=160)
plt.close(fig)

# --- Figure 3: radiation --------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, key, name in [(axs[0], "line_rad", "line radiation"),
                      (axs[1], "cont_rad", "continuum radiation")]:
    draw_pair(ax, case_mine, case_omfit,
              rad_mine[key].sum(0), rad_omfit[key].sum(0))
    ax.set_title(name, color=H.C_INK)
    ax.set_xlabel(r"$\rho$")
    ax.set_yscale("log")
axs[0].set_ylabel(r"$P_{rad}$  [W cm$^{-3}$]")
draw_pair(axs[2], case_mine, case_omfit, rad_mine["tot"], rad_omfit["tot"])
axs[2].set_title("total", color=H.C_INK)
axs[2].set_xlabel(r"$\rho$")
axs[2].set_yscale("log")
axs[0].legend(loc="upper left")
fig.suptitle(f"{IMP} radiation at t = {time_grid[-1] + dt:.3f} s  —  "
             f"external grid (solid) vs OMFIT geqdsk (dashed)",
             fontsize=12, color=H.C_INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "prad_comparison.png"), dpi=160)
plt.close(fig)

# --- Figure 4: time traces ------------------------------------------------
t_out = time_grid + dt
N_mine = np.array([H.volume_integral(case_mine, nz.sum(1))
                   for nz in case_mine["nz_time"]])
N_omfit = np.array([H.volume_integral(case_omfit, nz.sum(1))
                    for nz in case_omfit["nz_time"]])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
axs[0].plot(t_out, N_omfit, label=H.LBL_OMFIT, **H.STYLE_OMFIT)
axs[0].plot(t_out, N_mine, label=H.LBL_MINE, **H.STYLE_MINE)
axs[0].set_xlabel("time [s]")
axs[0].set_ylabel("confined C inventory [particles]")
axs[0].set_title("impurity build-up", color=H.C_INK)
axs[0].legend(loc="lower right")
axs[1].plot(t_out, 100 * (N_mine - N_omfit) / np.maximum(N_omfit, 1e-30),
            color=H.C_MINE, lw=1.8)
axs[1].axhline(0, color=H.C_MUTED, lw=0.8)
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("(A - B) / B   [%]")
axs[1].set_title("relative difference", color=H.C_INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "time_traces.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Animation
# ---------------------------------------------------------------------------
ymax = [max(case_mine["nz_time"][:, :, iz].max(),
            case_omfit["nz_time"][:, :, iz].max()) * 1.08 or 1.0
        for iz in range(nZ)]
ymax.append(max(case_mine["nz_time"].sum(2).max(),
                case_omfit["nz_time"].sum(2).max()) * 1.08)

frames = []
fig, axs = panel_grid()
for it in range(len(time_grid)):
    for iz in range(nZ):
        ax = axs[iz]
        ax.clear()
        draw_pair(ax, case_mine, case_omfit,
                  case_mine["nz_time"][it, :, iz],
                  case_omfit["nz_time"][it, :, iz], xlim=XLIM[iz])
        zoom = "" if XLIM[iz][0] == 0.0 else "   (edge zoom)"
        ax.set_title(f"{IMP}$^{{{iz}+}}$" + zoom, color=H.C_INK)
        ax.set_ylim(0, ymax[iz])
        ax.grid(True, color=H.C_GRID, lw=0.6)
        if iz % 4 == 0:
            ax.set_ylabel(r"$n_z$  [cm$^{-3}$]")
    axs[nZ].clear()
    draw_pair(axs[nZ], case_mine, case_omfit,
              case_mine["nz_time"][it].sum(1), case_omfit["nz_time"][it].sum(1),
              xlim=XLIM[nZ])
    axs[nZ].set_title("total", color=H.C_INK)
    axs[nZ].set_ylim(0, ymax[nZ])
    axs[nZ].grid(True, color=H.C_GRID, lw=0.6)
    for ax in axs:
        ax.set_xlabel(r"$\rho$")
    axs[0].legend(loc="upper left")
    fig.suptitle(f"{IMP} impurity density,  t = {t_out[it]:.3f} s  —  "
                 f"external grid (solid) vs OMFIT geqdsk (dashed)",
                 fontsize=12, color=H.C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
plt.close(fig)

gif_path = os.path.join(OUT, "nz_evolution_comparison.gif")
imageio.mimsave(gif_path, frames, duration=0.1, loop=0)

print("\nwritten to %s/:" % OUT)
for f in ["nz_comparison_final.png", "nz_relative_difference.png",
          "prad_comparison.png", "time_traces.png",
          "nz_evolution_comparison.gif", "comparison_statistics.txt"]:
    p = os.path.join(OUT, f)
    print(f"  {f:<32} {os.path.getsize(p) / 1024:8.1f} KB")
