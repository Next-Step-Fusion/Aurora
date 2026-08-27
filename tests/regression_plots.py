"""Optional figures for the regression suite.

Enabled with ``pytest tests/test_regression.py --plot``; nothing here runs
otherwise. Every figure follows the same convention as
``tests/test_with_omfit.py``:

    solid, thick, semi-transparent  =  the current run
    dashed, thin, drawn on top      =  the stored baseline

so that a perfect match still shows the dashes over the solid band, and any
regression appears as a visible separation.

Figures are written before the assertions run, so a *failing* test still leaves
the picture that shows what moved.
"""

import os

import numpy as np

import helpers as H

# Reuse the validated two-colour scheme; only the roles are renamed.
C_NOW, C_REF = H.C_MINE, H.C_OMFIT
STYLE_NOW = dict(H.STYLE_MINE)
STYLE_REF = dict(H.STYLE_OMFIT)
LBL_NOW, LBL_REF = "current run", "baseline"

MAX_PANELS = 7          # charge states shown individually (W has 75)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H.apply_style()
    return plt


def outdir(path=None):
    path = path or os.environ.get("AURORA_PLOT_DIR", "outputs")
    os.makedirs(path, exist_ok=True)
    return path


def _select_states(nz):
    """Charge states worth a panel: the strongest ones, in ascending order.

    ``nz`` is (nt, nZ, nr). W has 75 charge states; showing all of them is
    unreadable, so the ones carrying the density are picked.
    """
    peak = np.nanmax(np.abs(nz), axis=(0, 2))
    n = min(MAX_PANELS, len(peak))
    idx = np.argsort(peak)[::-1][:n]
    return np.sort(idx)


def _pair(ax, x_ref, y_ref, x_now, y_now, xlim=None):
    ax.plot(x_ref, y_ref, label=LBL_REF, **STYLE_REF)
    ax.plot(x_now, y_now, label=LBL_NOW, **STYLE_NOW)
    ax.axvline(1.0, color=H.C_MUTED, lw=0.7, ls=":", zorder=1)
    if xlim:
        ax.set_xlim(*xlim)


def _panels(plt, n, figsize=(15, 7.5)):
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(figsize[0],
                                                 figsize[1] * nrow / 2))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs


# ---------------------------------------------------------------------------
# nz / radiation profiles
# ---------------------------------------------------------------------------
def plot_profiles(name, ref, now, out, imp, it=-1):
    """nz per charge state at one time slice, plus the summed total."""
    plt = _plt()
    states = _select_states(now["nz"])
    fig, axs = _panels(plt, len(states) + 1)
    t = float(np.atleast_1d(now["t_slices"])[it])

    for k, z in enumerate(states):
        xlim = H.auto_xlim([
            (ref["rhop"], np.abs(ref["nz"][:, z, :]).max(0)),
            (now["rhop"], np.abs(now["nz"][:, z, :]).max(0)),
        ], hard_max=1.05)
        _pair(axs[k], ref["rhop"], ref["nz"][it, z], now["rhop"],
              now["nz"][it, z], xlim=xlim)
        zoom = "" if xlim[0] == 0.0 else "  (edge zoom)"
        axs[k].set_title(f"{imp}$^{{{z}+}}$" + zoom, color=H.C_INK)
        axs[k].set_ylim(bottom=0)
        axs[k].set_xlabel(r"$\rho$")
    axs[0].set_ylabel(r"$n_z$  [cm$^{-3}$]")
    axs[0].legend(loc="upper left")

    a = axs[len(states)]
    _pair(a, ref["rhop"], ref["nz"][it].sum(0), now["rhop"],
          now["nz"][it].sum(0), xlim=(0, 1.05))
    a.set_title("total (all charge states)", color=H.C_INK)
    a.set_ylim(bottom=0)
    a.set_xlabel(r"$\rho$")

    fig.suptitle(f"{name}: impurity density at t = {t:.4f} s   "
                 f"—  current (solid) vs baseline (dashed)",
                 fontsize=12, color=H.C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(out, f"nz_{name}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_radiation(name, ref, now, out, it=-1):
    """Line, continuum and total radiation summed over charge states."""
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, key, title in ((axs[0], "line_rad", "line radiation"),
                           (axs[1], "cont_rad", "continuum radiation"),
                           (axs[2], None, "total")):
        if key is None:
            yr = ref["line_rad"][it].sum(0) + ref["cont_rad"][it].sum(0)
            yn = now["line_rad"][it].sum(0) + now["cont_rad"][it].sum(0)
        else:
            yr, yn = ref[key][it].sum(0), now[key][it].sum(0)
        _pair(ax, ref["rhop"], np.maximum(yr, 1e-30),
              now["rhop"], np.maximum(yn, 1e-30), xlim=(0, 1.05))
        ax.set_title(title, color=H.C_INK)
        ax.set_xlabel(r"$\rho$")
        ax.set_yscale("log")
    axs[0].set_ylabel(r"$P_{rad}$  [W cm$^{-3}$]")
    axs[0].legend(loc="lower left")
    t = float(np.atleast_1d(now["t_slices"])[it])
    fig.suptitle(f"{name}: radiation at t = {t:.4f} s   "
                 f"—  current (solid) vs baseline (dashed)",
                 fontsize=12, color=H.C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, f"prad_{name}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_traces(name, ref, now, out):
    """Volume-integrated inventory and radiated power over the full time grid."""
    plt = _plt()
    t = now["time"] if "time" in now else np.arange(now["n_conf"].shape[0])
    tr = ref["time"] if "time" in ref.files else t
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0))

    axs[0].plot(tr, ref["n_conf"].sum(1), label=LBL_REF, **STYLE_REF)
    axs[0].plot(t, now["n_conf"].sum(1), label=LBL_NOW, **STYLE_NOW)
    axs[0].set_ylabel("confined inventory [particles]")
    axs[0].set_title("impurity build-up", color=H.C_INK)
    axs[0].legend(loc="lower right")

    for ax, key, title in ((axs[1], "p_line", "line radiated power"),
                           (axs[2], "p_cont", "continuum radiated power")):
        ax.plot(tr, ref[key], label=LBL_REF, **STYLE_REF)
        ax.plot(t, now[key], label=LBL_NOW, **STYLE_NOW)
        ax.set_ylabel("P [W]")
        ax.set_title(title, color=H.C_INK)
    for ax in axs:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"{name}: time traces  —  current (solid) vs baseline (dashed)",
                 fontsize=12, color=H.C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, f"traces_{name}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def animate_profiles(name, ref, now, out, imp):
    """Animate the frozen time slices, baseline vs current.

    Only the slices kept in the baseline are available (``N_SLICES`` in
    test_regression.py), so this is a short sequence, not a smooth movie.
    """
    import imageio

    plt = _plt()
    states = _select_states(now["nz"])
    nt = now["nz"].shape[0]

    xlims = [H.auto_xlim([
        (ref["rhop"], np.abs(ref["nz"][:, z, :]).max(0)),
        (now["rhop"], np.abs(now["nz"][:, z, :]).max(0)),
    ], hard_max=1.05) for z in states]
    ymax = [1.08 * max(np.nanmax(ref["nz"][:, z, :]),
                       np.nanmax(now["nz"][:, z, :])) or 1.0 for z in states]
    ymax.append(1.08 * max(np.nanmax(ref["nz"].sum(1)),
                           np.nanmax(now["nz"].sum(1))))

    fig, axs = _panels(plt, len(states) + 1)
    frames = []
    for it in range(nt):
        for k, z in enumerate(states):
            axs[k].clear()
            _pair(axs[k], ref["rhop"], ref["nz"][it, z], now["rhop"],
                  now["nz"][it, z], xlim=xlims[k])
            zoom = "" if xlims[k][0] == 0.0 else "  (edge zoom)"
            axs[k].set_title(f"{imp}$^{{{z}+}}$" + zoom, color=H.C_INK)
            axs[k].set_ylim(0, ymax[k])
            axs[k].set_xlabel(r"$\rho$")
            axs[k].grid(True, color=H.C_GRID, lw=0.6)
        axs[0].set_ylabel(r"$n_z$  [cm$^{-3}$]")
        axs[0].legend(loc="upper left")
        a = axs[len(states)]
        a.clear()
        _pair(a, ref["rhop"], ref["nz"][it].sum(0), now["rhop"],
              now["nz"][it].sum(0), xlim=(0, 1.05))
        a.set_title("total (all charge states)", color=H.C_INK)
        a.set_ylim(0, ymax[-1])
        a.set_xlabel(r"$\rho$")
        a.grid(True, color=H.C_GRID, lw=0.6)
        t = float(np.atleast_1d(now["t_slices"])[it])
        fig.suptitle(f"{name}: impurity density, t = {t:.4f} s   "
                     f"—  current (solid) vs baseline (dashed)",
                     fontsize=12, color=H.C_INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    p = os.path.join(out, f"nz_evolution_{name}.gif")
    imageio.mimsave(p, frames, duration=0.6, loop=0)
    return p


# ---------------------------------------------------------------------------
# FACIT
# ---------------------------------------------------------------------------
def _spread_states(y):
    """A spread of charge states, not just the strongest.

    ``y`` is (nr, nZ). Selecting the top-N by magnitude tends to return
    neighbouring states with nearly identical curves; this picks states evenly
    spaced among those that carry signal, so the panels differ from each other.
    """
    peak = np.nanmax(np.abs(y), axis=0)
    live = np.where(peak > 1e-6 * np.nanmax(peak))[0]
    live = live[live > 0]
    if len(live) == 0:
        return np.array([1])
    n = min(MAX_PANELS - 2, len(live))
    return np.unique(live[np.linspace(0, len(live) - 1, n).astype(int)])


def plot_facit(name, ref, now, out, imp):
    """Neoclassical Dz and Vconv, one column per charge state.

    Small multiples rather than an overlay: the coefficients of neighbouring
    charge states differ by orders of magnitude near the edge and overlap
    almost exactly in the core, so a single axis is unreadable either way.
    ``Vconv`` uses a symmetric-log scale -- the separatrix spike is ~100x the
    core value and would otherwise flatten everything inside r/a < 0.95.
    """
    plt = _plt()
    states = _spread_states(now["Dz"])
    ncol = len(states)
    fig, axs = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.4), squeeze=False)

    vmax = np.nanmax(np.abs(now["Vconv"][:, states]))
    lin = max(vmax * 1e-3, 1e-6)
    for k, z in enumerate(states):
        a0, a1 = axs[0][k], axs[1][k]
        a0.plot(ref["roa"], ref["Dz"][:, z], label=LBL_REF, **STYLE_REF)
        a0.plot(now["roa"], now["Dz"][:, z], label=LBL_NOW, **STYLE_NOW)
        a0.set_yscale("log")
        a0.set_title(f"{imp}$^{{{z}+}}$", color=H.C_INK)

        a1.plot(ref["roa"], ref["Vconv"][:, z], **STYLE_REF)
        a1.plot(now["roa"], now["Vconv"][:, z], **STYLE_NOW)
        a1.set_yscale("symlog", linthresh=lin)
        a1.axhline(0, color=H.C_MUTED, lw=0.8)
        a1.set_xlabel(r"$r/a$")
        for a in (a0, a1):
            a.set_xlim(0, 1.0)
    axs[0][0].set_ylabel(r"$D_z$  [m$^2$/s]")
    axs[1][0].set_ylabel(r"$V_{conv}$  [m/s]  (symlog)")
    axs[0][0].legend(loc="upper right")
    fig.suptitle(f"{name}: FACIT neoclassical coefficients  —  "
                 f"current (solid) vs baseline (dashed)",
                 fontsize=12, color=H.C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, f"facit_{name}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_multi_species(ref, now, out, impurities):
    """Shared Zeff and each species' contribution, plus its FACIT D/V."""
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

    x_ref, x_now = ref["rhop"], now["rhop"]
    axs[0].plot(x_ref, ref["Zeff"], **STYLE_REF)
    axs[0].plot(x_now, now["Zeff"], **STYLE_NOW)
    axs[0].axvline(1.0, color=H.C_MUTED, lw=0.7, ls=":", zorder=1)
    axs[0].set_ylabel(r"$Z_{eff}$")
    axs[0].set_xlabel(r"$\rho_{pol}$")
    axs[0].set_title("shared effective charge", color=H.C_INK)
    axs[0].plot([], [], label=LBL_REF, **STYLE_REF)
    axs[0].plot([], [], label=LBL_NOW, **STYLE_NOW)
    axs[0].legend(loc="upper left")

    for imp in impurities:
        axs[1].plot(x_ref, ref[f"dZeff_{imp}"], **STYLE_REF)
        axs[1].plot(x_now, now[f"dZeff_{imp}"], **STYLE_NOW)
        k = int(np.argmax(now[f"dZeff_{imp}"]))
        axs[1].annotate(imp, (x_now[k], now[f"dZeff_{imp}"][k]),
                        fontsize=10, color=H.C_INK,
                        xytext=(4, 4), textcoords="offset points")
    axs[1].axvline(1.0, color=H.C_MUTED, lw=0.7, ls=":", zorder=1)
    axs[1].set_ylabel(r"$\Delta Z_{eff}$ contribution")
    axs[1].set_xlabel(r"$\rho_{pol}$")
    axs[1].set_title("per-species contribution", color=H.C_INK)

    for imp in impurities:
        peak = np.nanmax(np.abs(now[f"{imp}_Dz_si"]), axis=0)
        z = int(np.argmax(peak))
        roa = now[f"{imp}_roa"]
        axs[2].plot(ref[f"{imp}_roa"], ref[f"{imp}_Dz_si"][:, z], **STYLE_REF)
        axs[2].plot(roa, now[f"{imp}_Dz_si"][:, z], **STYLE_NOW)
        k = int(np.argmax(now[f"{imp}_Dz_si"][:, z]))
        axs[2].annotate(f"{imp}$^{{{z}+}}$", (roa[k], now[f"{imp}_Dz_si"][k, z]),
                        fontsize=10, color=H.C_INK,
                        xytext=(4, 4), textcoords="offset points")
    axs[2].set_ylabel(r"$D_z$  [m$^2$/s]")
    axs[2].set_xlabel(r"$r/a$")
    axs[2].set_yscale("log")
    axs[2].set_title("FACIT $D_z$, strongest charge state", color=H.C_INK)

    fig.suptitle("multi-species (C + W, shared $Z_{eff}$)  —  "
                 "current (solid) vs baseline (dashed)",
                 fontsize=12, color=H.C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, "multi_species.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
