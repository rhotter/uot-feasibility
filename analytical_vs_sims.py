"""
Compare analytical solution to MCX simulations for a semi-infinite medium
with and without reflection. This is a replication of Figure 5e from 
https://pmc.ncbi.nlm.nih.gov/articles/PMC2863034
"""

# %% Import modules
import numpy as np
import pmcx
from matplotlib import pyplot as plt

# %% Define diffusion function
def cwdiffusion(mua, musp, Reff, srcpos, detpos):
    """
    Semi-infinite medium analytical solution to diffusion model
    Args:
        mua: absorption coefficient (1/mm)
        musp: reduced scattering coefficient (1/mm)
        Reff: effective reflection coefficient
        srcpos: source position array (x,y,z)
        detpos: detector positions array (x,y,z)
    Returns:
        phi: diffusion profile
        r: (optional) source detector separations
    """
    D = 1/(3*(mua + musp))
    zb = (1 + Reff) / (1 - Reff) * 2 * D
    
    z0 = 1 / (musp + mua)  # Note: order switched to match MATLAB
    
    # Create virtual source positions
    src_z0 = np.array([srcpos[0], srcpos[1], srcpos[2] + z0])
    src_image = np.array([srcpos[0], srcpos[1], srcpos[2] - z0 - 2*zb])
    
    # Calculate distances
    r1 = np.sqrt(np.sum((detpos - src_z0)**2, axis=1))
    r2 = np.sqrt(np.sum((detpos - src_image)**2, axis=1))
    
    # Calculate fluence
    b = np.sqrt(3 * mua * musp)  # Different from previous mueff calculation
    phi = 1/(4*np.pi*D) * (np.exp(-b*r1)/r1 - np.exp(-b*r2)/r2)
    
    return phi

# %% Set up simulation parameters
g = 0.01               # anisotropy factor (matching the MATLAB example)
mua = 0.005           # absorption coefficient [1/mm]
mus = 1.0             # scattering coefficient [1/mm]

def build_cfg(isreflect):
    return {
        'nphoton': 3e7,
        'vol': np.ones([60, 60, 60], dtype='uint8'),
        'tstart': 0,
        'tend': 5e-9,
        'tstep': 1e-10,
        'srcpos': [30, 30, 0],
        'srcdir': [0, 0, 1],
        'prop': [
            [0, 0, 1, 1],               # medium 0: environment
            [mua, mus, g, 1.37 if isreflect else 1]       # medium 1: cube
        ],
        'isreflect': isreflect
    }

# %% Run simulations
# Run simulation without reflection
print('Running simulation without reflection...')
cfg1 = build_cfg(0)
res1 = pmcx.mcxlab(cfg1)

# Run simulation with reflection
print('Running simulation with reflection...')
cfg2 = build_cfg(1)
res2 = pmcx.mcxlab(cfg2)

# %% Process results
# Calculate CW fluence by summing over time
cwf1 = np.sum(res1['flux'], axis=3)
cwf2 = np.sum(res2['flux'], axis=3)

# Prepare data for plotting
x = np.arange(30)
srcpos = np.array([0, 0, 0])
detpos = np.column_stack((x + 0.5, np.full_like(x, 0.5), np.full_like(x, 0.5)))

# Calculate diffusion solutions
musp = mus * (1 - g)
phi_no_refl = cwdiffusion(mua, musp, 0, srcpos, detpos)
phi_refl = cwdiffusion(mua, musp, 0.493, srcpos, detpos)

# %% Plot results
plt.figure(figsize=(10, 6))
plt.semilogy(x + 1, phi_no_refl, 'r-', label='Diffusion (no reflection)')
plt.semilogy(x + 1, cwf1[30, 30:, 0] * cfg1['tstep'], 'o', label='MCX (no reflection)')
plt.semilogy(x + 1, phi_refl, 'r--', label='Diffusion (with reflection)')
plt.semilogy(x + 1, cwf2[30, 30:, 0] * cfg2['tstep'], '+', label='MCX (with reflection)')

plt.xlabel('x (mm)')
plt.ylabel('Fluence rate in 1/(mm² s)')
plt.legend(frameon=False)
plt.grid(True)
plt.title('Spatial Decay Profile (y=30, z=0)')
plt.show()
# %%
