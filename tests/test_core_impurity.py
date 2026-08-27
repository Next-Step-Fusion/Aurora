import numpy as np
import aurora
import matplotlib.pyplot as plt
import os
import h5py
import imageio

namelist = aurora.load_default_namelist()

h5_file = h5py.File("tests/data/centaur.h5", "r")
state = h5_file['history/state']
idt = -1 # last timestep

rho_tor = state["normalized_rho"][idt]
psi_profile = state["psi_profile"][idt] # [Wb] - psi(rho_tor)
psi_N = (psi_profile - psi_profile[0]) / (psi_profile[-1] - psi_profile[0]) # - psi_N(rho_tor)
rho_pol = np.sqrt(psi_N) # - rho_pol(rho_tor)

psi_axis = state["psi_axis"][idt] # [Wb]
psi_separatrix = state["psi_separatrix"][idt] # [Wb] - psi_N = 1
psi_boundary = state["psi_boundary"][idt] # [Wb] - psi_N = 0.995
psi_RZ = state["psi"][idt] # [Wb] - psi(R,Z)
R = state["x_psi"][idt] # [m]
Z = state["y_psi"][idt] # [m] 
q = state["q"][idt] # - q(rho_tor)
aminor = state["minor_radius"][idt] # [m]
volume = state["volume"][idt] # [m^3]
geqdsk_data = {
    "comment" : f"{999}",
    "shot"    : 0,
    "nx"      : len(R),
    "ny"      : len(Z),
    "rdim"    : state["eqdsk_plasma_equilibrium_data"][idt, 0],
    "zdim"    : state["eqdsk_plasma_equilibrium_data"][idt, 1],
    "rcentr"  : state["eqdsk_plasma_equilibrium_data"][idt, 2],
    "rleft"   : state["eqdsk_plasma_equilibrium_data"][idt, 3],
    "zmid"    : state["eqdsk_plasma_equilibrium_data"][idt, 4],
    "rmagx"   : state["eqdsk_plasma_equilibrium_data"][idt, 5],
    "zmagx"   : state["eqdsk_plasma_equilibrium_data"][idt, 6],
    "simagx"  : state["eqdsk_plasma_equilibrium_data"][idt, 7],
    "sibdry"  : state["eqdsk_plasma_equilibrium_data"][idt, 8],
    "bcentr"  : state["eqdsk_plasma_equilibrium_data"][idt, 9],
    "cpasma"  : state["eqdsk_plasma_equilibrium_data"][idt, 10],
    "fpol"    : state["eqdsk_data_fpol_out"][idt],
    "pres"    : state["eqdsk_data_pres_out"][idt],
    "ffprime" : state["eqdsk_data_ffprime_out"][idt],
    "pprime"  : state["eqdsk_data_pprime_out"][idt],
    "psi"     : psi_RZ / (2 * np.pi),  # [Wb / radian]
    "qpsi"    : state["eqdsk_data_qpsi_out"][idt],
    "nbdry"   : len(state["boundary"][idt]) - 1,  # jbound = num_b -1
    "rbdry"   : state["boundary"][idt, :, 0][: len(state["boundary"][idt]) - 1],  # jbound = num_b -1
    "zbdry"   : state["boundary"][idt, :, 1][: len(state["boundary"][idt]) - 1],  # jbound = num_b -1
    "nlim": 21,
    "rlim": np.array([1.26, 1.44999999999999, 2.17799999999999, 2.17799999999999, 2.28299999999999, 2.46303142, 2.7, 2.67399999999999, 2.6, 2.72999999999999, 2.72999999999999, 2.6, 2.67399999999999, 2.7, 2.46303142, 2.28299999999999, 2.17799999999999, 2.17799999999999, 1.44999999999999, 1.26, 1.26]),
    "zlim": np.array([-0.27, -0.689999999999999, -1.19999999999999, -1.29299999999999, -1.56, -1.37471264, -1.41999999999999, -1.23749999999999, -1.1, -0.45517241, 0.45517241, 1.1, 1.23749999999999, 1.41999999999999, 1.37471264, 1.56, 1.29299999999999, 1.19999999999999, 0.689999999999999, 0.27, -0.27])
}
rvol_lcfs = np.sqrt(volume[-1] / (2 * np.pi**2 * geqdsk_data["rcentr"]))

from freeqdsk import geqdsk
with open("tests/data/centaur.geqdsk", "w") as fid:
    geqdsk.write(geqdsk_data, fid)

ne = state["electron_density_profile"][idt] # [1e19 m^-3] - ne(rho_tor)
Te = state["electron_temperature_profile"][idt] # [eV] - Te(rho_tor)
Ti = state["ion_temperature_profile"][idt] # [eV] - Ti(rho_tor)
n0 = 0.001 * np.exp((rho_tor - 1.0) / 0.02) # [1e19 m^-3]

# Setup namelist
dt = 1e-3
time_grid = np.arange(0.0, 0.1, dt)
nz = np.zeros((482, 7))
nz_time = []

for it in range(len(time_grid)):
    namelist["timing"] = {
        "dt_increase": np.array([1.0, 1.0]),
        "dt_start": np.array([dt, dt]),
        "steps_per_cycle": np.array([1, 1]),
        "times": np.array([time_grid[it], time_grid[it] + dt]),
    }
    namelist["Baxis"] = geqdsk_data["bcentr"]
    namelist["Raxis_cm"] = geqdsk_data["rmagx"] * 1.0e2  # [cm]
    namelist["rvol_lcfs"] = rvol_lcfs * 1.0e2  # [m] --> [cm]

    namelist['kin_profs']['ne']['rhop'] = rho_pol
    namelist['kin_profs']['ne']['vals'] = ne * 1e13 # [1e19 m^-3] --> [cm^-3]

    namelist['kin_profs']['Te']['rhop'] = rho_pol
    namelist['kin_profs']['Te']['vals'] = Te

    namelist['kin_profs']['Ti']['rhop'] = rho_pol
    namelist['kin_profs']['Ti']['vals'] = Ti

    namelist['kin_profs']['n0']['rhop'] = rho_pol
    namelist['kin_profs']['n0']['vals'] = n0 * 1e13 # [1e19 m^-3] --> [cm^-3]

    namelist['imp'] = 'C'
    namelist['source_type'] = 'const'
    namelist['source_rate'] = 1e21 # [particles/s]

    namelist['LBO'] = None # {'n_particles': 0.0, 't_fall': 1e99, 't_rise': 1e99, 't_start': 1e99}

    asim = aurora.aurora_sim(namelist)

    D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
    V_z = np.zeros(len(asim.rvol_grid))

    out = asim.run_aurora(D_z, V_z, nz_init=nz)
    # print(out['nz'].shape)
    nz = np.maximum(out['nz'][:, :, -1], 0.0)
    print(it, nz[0, 3])
    nz_time.append(nz)
    reservoirs = asim.reservoirs_time_traces(plot=False)
    # print(reservoirs)



    fig, axs = plt.subplots(nrows=nz.shape[1], ncols=1, figsize=(6, 4 * nz.shape[1]))
    for iz in range(nz.shape[1]):
        axs[iz].plot(asim.rhop_grid, nz[:, iz])
        axs[iz].set_xlabel('rhop')
        axs[iz].set_ylabel(f'nz[{iz}] [cm^-3]')
        axs[iz].set_title(f'C{iz}+')
        axs[iz].axhline(0, color='k', linestyle='--')
    fig.tight_layout()
    fig.savefig('impurity_density_profiles.png', dpi=300)
    plt.close(fig)


frames = []
fig, ax = plt.subplots(figsize=(8, 6))

for i, nz in enumerate(nz_time):
    ax.clear()
    for iz in range(nz.shape[1]):
        ax.plot(asim.rhop_grid, nz[:, iz], label=f'C{iz}+')
    ax.set_xlabel('rhop')
    ax.set_ylabel('nz [cm^-3]')
    ax.set_title(f'Impurity Density Profiles at t={time_grid[i]:.4f}s')
    ax.legend()
    ax.set_ylim([0, np.max(nz_time)])
    
    # Save frame
    frame_path = f'/tmp/frame_{i:04d}.png'
    fig.savefig(frame_path, dpi=100, bbox_inches='tight')
    frames.append(imageio.imread(frame_path))
    os.remove(frame_path)

plt.close(fig)

# Save as gif
imageio.mimsave('impurity_evolution.gif', frames, duration=0.1)