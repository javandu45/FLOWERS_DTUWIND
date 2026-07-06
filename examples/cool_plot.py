# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from FLOWERS import NOJ_flowers, gaussian_flowers, TurbOPark_flowers, fuga_flowers
from utils import generic_site

wind_farm = "Thor"

turbine = V80()

D = turbine.diameter()  # Turbine diameter in meters
fontsize = 14

def plot_9th_turbine_aep_impact(proxy, ax, grid_resolution=50):
    """
    Plot AEP changes when adding a 9th turbine to 8 turbines in a square layout.

    Args:
        proxy: The wind farm proxy object with calculate_aep method
        ax: Matplotlib axes object to plot on
        grid_resolution: Number of points along each axis for the grid

    Returns:
        contourf: The contourf object for creating a shared colorbar
    """

    # Define 8 turbines in a square layout
    base_layout = np.array([
        [0, 0],
        [7*D, 0],
        [14*D, 0],
        [0, 7*D],
        [14*D, 7*D],
        [0, 14*D],
        [7*D, 14*D],
        [14*D, 14*D]
    ])

    # Calculate baseline AEP with 8 turbines
    baseline_aep = proxy.aep(base_layout[:, 0], base_layout[:, 1])

    # Get turbine diameter and calculate minimum spacing (2D)
    turbine_diameter = D
    min_spacing = 2 * turbine_diameter

    # Create grid for 9th turbine positions
    x_range = np.linspace(0*D, 14*D, grid_resolution)
    y_range = np.linspace(0*D, 14*D, grid_resolution)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate AEP for each position of 9th turbine
    aep_values = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            x_9th = X[i, j]
            y_9th = Y[i, j]

            # Check minimum spacing constraint
            distances = np.sqrt((base_layout[:, 0] - x_9th)**2 + (base_layout[:, 1] - y_9th)**2)
            if np.any(distances < min_spacing):
                aep_values[i, j] = np.nan  # Invalid position
                continue

            x_all = np.append(base_layout[:, 0], x_9th)
            y_all = np.append(base_layout[:, 1], y_9th)
            aep_values[i, j] = proxy.aep(x_all, y_all) - baseline_aep

    max_aep = np.nanmax(aep_values)
    if max_aep > 0:
        aep_values = aep_values / max_aep

    # Create plot on the provided axes
    contourf = ax.contourf(X/D, Y/D, aep_values, levels=50, cmap='RdYlGn')
    contour = ax.contour(X/D, Y/D, aep_values, levels=50, colors='black', linewidths=0.5, alpha=1)

    # Plot existing 8 turbines (normalized)
    ax.scatter(base_layout[:, 0]/D, base_layout[:, 1]/D,
                c='black', s=500, marker='o',
                edgecolors='black', linewidths=5,
                label='Existing Turbines', zorder=5)

    ax.set_xlabel('X/$D_{WT}$', fontsize=fontsize)
    ax.set_ylabel('Y/$D_{WT}$', fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 7, 14])
    ax.set_yticks([0, 7, 14])
    ax.tick_params(labelsize=fontsize)
    ax.set_aspect('equal')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)

    return contourf

# %%

site = generic_site(10)

wfm = fuga_flowers(site=site, windTurbines=turbine, n_terms=10, dx=None)
wfm = gaussian_flowers(site=site, windTurbines=turbine, n_terms=10)

site = generic_site(10)

# Create figure with 7 subplots
fig, axes = plt.subplots(1, 1, figsize=(8,8), dpi=200)

contourf = None
# Plot AEP impact for each proxy and get contourf object
contourf = plot_9th_turbine_aep_impact(wfm, axes, grid_resolution=100)

# Get legend from first subplot
handles, labels = axes.get_legend_handles_labels()

# Create shared colorbar
plt.tight_layout()
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contourf, cax=cbar_ax, format="%.3f")
cbar.set_label('Normalized AEP ($AEP/AEP_{max}$)', fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)
fig.legend(handles, labels, fontsize=fontsize, ncol=1, loc='lower center', bbox_to_anchor=(0.5, -0.03))
plt.show()
