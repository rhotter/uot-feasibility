# %% [markdown]
# # Ultrasound-Modulated Optical Tomography (UOT) Feasibility Analysis
#
# This notebook analyzes the feasibility of UOT for detecting neural activity by:
# 1. Estimating the number of tagged photons
# 2. Analyzing contrast-to-noise requirements
# 3. Simulating detection depth vs. absorption contrast

# %% [markdown]
# ## Setup and Constants

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# %% [markdown]
# ## Core Physics Parameters

# %%
# Tissue optical properties
ABSORPTION_COEF = 0.02  # Absorption coefficient [mm^-1]
REDUCED_SCATTERING_COEF = 0.67  # Reduced scattering coefficient [mm^-1]

# Light properties
WAVELENGTH = 800e-9  # Wavelength [m] (800 nm)
INCIDENT_INTENSITY = 100e-3  # Incident light intensity [W/cm^2]
SOURCE_AREA = 1  # Source area [cm^2]

# Ultrasound properties
TAGGING_EFFICIENCY = 0.1  # Fraction of light that gets frequency-shifted
FOCAL_SPOT_SIZE = 1  # Focal spot size [mm^2]

# Detection properties
DETECTOR_AREA = 100  # Detector area [mm^2]
DETECTOR_EFFICIENCY = 0.075  # Fraction of detected photons that produce signal
DETECTOR_COLLECTION = 0.1  # Fraction of light collected by detector

# Calculate source power and photon energy
SOURCE_POWER = INCIDENT_INTENSITY * SOURCE_AREA  # [W]
PHOTON_ENERGY = h * c / WAVELENGTH  # [J]
SOURCE_PHOTONS = SOURCE_POWER / PHOTON_ENERGY  # [photons/s]

# calculate effective attenuation coefficient
MU_EFF = np.sqrt(3 * ABSORPTION_COEF * (ABSORPTION_COEF + REDUCED_SCATTERING_COEF))


# %% [markdown]
# ## Optical Propagation Functions


# %%
def green_function_forward(z):
    """Calculate the forward Green's function for photon propagation."""
    z0 = -1 / MU_EFF
    return (
        -3
        * (MU_EFF + ABSORPTION_COEF)
        / (4 * np.pi)
        * 2
        * z
        * z0
        * (MU_EFF + 1 / z)
        * np.exp(-MU_EFF * z)
        / z**2
    )


def green_function_backward(z):
    """Calculate the backward Green's function for photon propagation."""
    return 2 * z / (4 * np.pi) * (MU_EFF + 1 / z) * np.exp(-MU_EFF * z) / z**2


def calc_tagged_photons(z):
    """Calculate the number of tagged photons reaching the detector."""
    forward_loss = green_function_forward(z) * TAGGING_EFFICIENCY * FOCAL_SPOT_SIZE
    backward_loss = green_function_backward(z) * DETECTOR_AREA

    return SOURCE_PHOTONS * forward_loss * backward_loss * DETECTOR_EFFICIENCY


def calc_cnr(n_tagged_baseline, n_tagged_contrast, n_untagged):
    """Calculate the contrast-to-noise ratio."""
    return (
        np.sqrt(2)
        * (n_tagged_contrast - n_tagged_baseline)
        / np.sqrt(n_tagged_baseline + n_tagged_contrast + n_untagged)
    )


# %% [markdown]
# ## Basic Photon Estimation Analysis


# %%
def run_basic_analysis(depth_mm=10):
    """Run a basic analysis of tagged photons at a specified depth."""
    # Calculate losses
    forward_loss = (
        green_function_forward(depth_mm) * TAGGING_EFFICIENCY * FOCAL_SPOT_SIZE
    )
    backward_loss = green_function_backward(depth_mm) * DETECTOR_AREA

    # Calculate detector power and number of tagged photons
    detected_power = SOURCE_POWER * forward_loss * backward_loss
    tagged_photons = detected_power / PHOTON_ENERGY

    # Display results
    print("=" * 50)
    print(f"{'Analysis at '+str(depth_mm)+'mm depth':<30}")
    print("=" * 50)
    print(f"{'Forward Loss':<30}: {forward_loss:.2e}")
    print(f"{'Backward Loss':<30}: {backward_loss:.2e}")
    print(f"{'Source Power':<30}: {SOURCE_POWER:.2e} W")
    print(f"{'Power at Detector':<30}: {detected_power:.2e} W")
    print(f"{'Source Photons/s':<30}: {SOURCE_PHOTONS:.2e}")
    print(f"{'Tagged Photons/s':<30}: {tagged_photons:.2e}")
    print("=" * 50)

    return tagged_photons


# Run the basic analysis
tagged_photons = run_basic_analysis(10)

# %% [markdown]
# ## Depth vs. Contrast Simulation


# %%
def calc_untagged_photons(filter_od):
    filter_transmission = 10.0 ** (-filter_od)
    n_untagged = (
        SOURCE_PHOTONS * DETECTOR_COLLECTION * filter_transmission * DETECTOR_EFFICIENCY
    )
    return n_untagged


def simulate_depth_vs_contrast():
    """Simulate and plot the relationship between detection depth and absorption contrast."""
    # Depth and contrast grids
    depths_mm = np.linspace(1, 100, 10000)
    contrasts = np.linspace(1.001, 2.5, 200)

    # Precompute tagged photons for all depths
    n_tagged_baseline = np.array([calc_tagged_photons(z) for z in depths_mm])

    def compute_depth_for_cnr1(filter_od):
        """Compute maximum depth for CNR=1 across different contrast values."""
        # Calculate untagged photon transmission
        n_untagged = calc_untagged_photons(filter_od)

        depths_cm = []
        for contrast in contrasts:
            # Calculate tagged photons with contrast
            n_tagged_contrast = n_tagged_baseline * contrast

            # Calculate CNR for each depth
            cnr_values = calc_cnr(n_tagged_baseline, n_tagged_contrast, n_untagged)

            # Find maximum depth where CNR >= 1
            valid_depths = depths_mm[cnr_values >= 1]
            max_depth_cm = (
                valid_depths.max() if valid_depths.size else 0
            ) / 10  # Convert to cm
            depths_cm.append(max_depth_cm)

        return np.array(depths_cm)

    # Plot setup
    filter_od_values = np.arange(3, 16, 1)
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=filter_od_values.min(), vmax=filter_od_values.max())

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot depth curves for each filter OD
    for filter_od in filter_od_values:
        depth_curve = compute_depth_for_cnr1(filter_od)
        ax.plot(contrasts, depth_curve, color=cmap(norm(filter_od)), linewidth=2)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(
        "Filter Optical Density (OD)", rotation=270, labelpad=20, fontsize=12
    )

    # Set labels and title
    ax.set_xlabel("Absorption Contrast", fontsize=12)
    ax.set_ylabel("Maximum Depth for CNR=1 (cm)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Detection Depth vs. Absorption Contrast for Various Filter ODs", fontsize=14
    )

    plt.tight_layout()
    plt.show()


# Run the simulation
simulate_depth_vs_contrast()

# %%
