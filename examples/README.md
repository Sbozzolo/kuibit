# Examples of working problems built with `kuibit`

In this directory, we collect working programs built with `kuibit`. You can use
these scripts as good examples of ``kuibit`` usage (or you can directly use
them).

> :warning: While `kuibit` is tested at each commit to ensure that nothing
>           breaks, these codes are not. If you find one that does not work,
>           please report it and we will fix that.

## Scripts available

### Utilities

| Name                           | Description                                                                                                                                                                                          |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `plot_ah_radius`               | Plots the coordinate radius of a given horizon as a function of time.                                                                                                                                |
| `plot_ah_separation`           | Plots the coordinate separation between the centroids of two given apparent horizons as a function of time.                                                                                          |
| `plot_constraints`             | Plots given reductions for the constraints (Hamiltonian, momentum, ...) as a function of time.                                                                                                       |
| `plot_em_energy`               | Plots the electromagnetic-wave luminosity and energy as measured by a given detector using Phi2 as a function of time.                                                                               |
| `plot_grid_var`                | Plots a 2D grid function among the ones output by Carpet on a given region. Optionally take logarithm or absolute value.                                                                             |
| `plot_gw_energy`               | Plots the gravitational-wave luminosity and energy as measured by a given detector as a function of time.                                                                                            |
| `plot_gw_linear_momentum`      | Plots the linear momentum lost by gravitational-wave as measured by a given detector as a function of time.                                                                                          |
| `plot_physical_time_per_hour`  | Plots the computational speed of the simulation by plotting how much physical time is simulated in one hour and one day.                                                                             |
| `plot_timeseries`              | Plots any timeseries among the ones output by IOASCII (e.g., scalars, reductions).                                                                                                                   |
| `plot_psi4_lm`                 | Plots the (l,m) mode of Psi4 as measured from a given detector.                                                                                                                                      |
| `plot_strain_lm`               | Plots the (l,m) mode of the gravitational-wave strain as measured from a given detector from Psi4.                                                                                                   |
| `plot_total_luminosity`        | Plots the combined electromagnetic and gravitational-wave luminosity as measured from a given detector with Phi2 and Psi4.                                                                           |
| `print_ah_formation_time`      | Prints the time when the given apparent horizon was first found.                                                                                                                                     |
| `print_available_iterations`   | Given a variable, prints which iterations are available across all the data and all the dimensions. Optionally, you can specify only one dimension. This script is useful for writing shell scripts. |
| `print_available_timeseries`   | Prints all the various timeseries found in the data (reductions, scalars, ...). This script is useful for exploring the data available in a simulation.                                              |
| `print_qlm_properties_at_time` | Prints some of the interesting properties (as from QuasiLocalMeasures) for a given horizon at a given time. Optionally estimates the Lorentz factor.                                                 |
| `save_reasampled_grid_data`    | Reads a grid function, resamples it to a given grid, and saves it to a file. This script is useful to move data from a cluster to your local machine (especially for 3D visualization)               |
