import numpy as np
import aurora
import matplotlib.pyplot as plt
from omfit_classes import omfit_eqdsk, omfit_gapy
import imageio
import os

namelist = aurora.load_default_namelist()
print(namelist)

geqdsk = omfit_eqdsk.OMFITgeqdsk('examples/example.gfile')
print(geqdsk)


ne = 10.0 * (1 - geqdsk["RHOVN"]**2) + 0.5 # [1e19 m^-3]
Te = 3.0 * (1 - geqdsk["RHOVN"]**2) + 0.1 # [keV]
Ti = 2.0 * (1 - geqdsk["RHOVN"]**2) + 0.1 # [keV]
n0 = 0.001 * np.exp((geqdsk["RHOVN"] - 1.0) / 0.02) # [1e19 m^-3]

# Setup namelist
dt = 1e-3
time_grid = np.arange(0.0, 0.1, dt)
nz = np.zeros((162, 7))
nz_time = []
for it in range(len(time_grid)):
    namelist["timing"] = {
        "dt_increase": np.array([1.0, 1.0]),
        "dt_start": np.array([dt, dt]),
        "steps_per_cycle": np.array([1, 1]),
        "times": np.array([time_grid[it], time_grid[it] + dt]),
    }
    namelist['kin_profs']['ne']['rhop'] = geqdsk["RHOVN"]
    namelist['kin_profs']['ne']['vals'] = ne * 1e13

    namelist['kin_profs']['Te']['rhop'] = geqdsk["RHOVN"]
    namelist['kin_profs']['Te']['vals'] = Te * 1e3

    namelist['kin_profs']['Ti']['rhop'] = geqdsk["RHOVN"]
    namelist['kin_profs']['Ti']['vals'] = Ti * 1e3

    namelist['kin_profs']['n0']['rhop'] = geqdsk["RHOVN"]
    namelist['kin_profs']['n0']['vals'] = n0 * 1e13

    namelist['imp'] = 'C'
    namelist['source_type'] = 'const'
    namelist['source_rate'] = 1e21 # [particles/s]

    namelist['LBO'] = None # {'n_particles': 0.0, 't_fall': 1e99, 't_rise': 1e99, 't_start': 1e99}

    asim = aurora.aurora_sim(namelist, geqdsk=geqdsk)

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