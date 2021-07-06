# Examples of working programs built with `kuibit`

In this directory, we collect working programs built with `kuibit`. You can use
these scripts as good examples of ``kuibit`` usage. These codes are ready to be
used, check out our
[recommendations](https://sbozzolo.github.io/kuibit/recommendation_examples.rst)
to get the most of out these examples.

> :warning: While `kuibit` is tested at each commit to ensure that nothing
>           breaks, these codes are not. If you find one that does not work,
>           please report it and we will fix that.

## Scripts available

You can achieve a lot using these scripts. We recommend downloading them and
placing them in you `$PATH` so that they can be used for your simulations.

### Plotting

| Name                          | Description                                                                                                                |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `plot_1d_vars`                | Plots one or more 1D grid functions output by Carpet. Optionally take logarithm and/or absolute value.                     |
| `plot_ah_coordinate_velocity` | Plots the coordinate velocities of a given horizon.                                                                        |
| `plot_ah_found`               | Plots the times at which the given apparent horizons were found.                                                           |
| `plot_ah_radius`              | Plots the coordinate radius of a given horizon as a function of time.                                                      |
| `plot_ah_separation`          | Plots the coordinate separation between the centroids of two given apparent horizons as a function of time.                |
| `plot_constraints`            | Plots given reductions for the constraints (Hamiltonian, momentum, ...) as a function of time.                             |
| `plot_em_energy`              | Plots the electromagnetic-wave luminosity and energy as measured by a given detector using Phi2 as a function of time.     |
| `plot_grid_var`               | Plots a 2D grid function among the ones output by Carpet on a given region. Optionally take logarithm or absolute value.   |
| `plot_gw_energy`              | Plots the gravitational-wave luminosity and energy as measured by a given detector as a function of time.                  |
| `plot_gw_angular_momentum`    | Plots the angular momentum lost by gravitational-wave as measured by a given detector as a function of time.               |
| `plot_gw_linear_momentum`     | Plots the linear momentum lost by gravitational-wave as measured by a given detector as a function of time.                |
| `plot_physical_time_per_hour` | Plots the computational speed of the simulation by plotting how much physical time is simulated in one hour and one day.   |
| `plot_timeseries`             | Plots any timeseries among the ones output by IOASCII (e.g., scalars, reductions).                                         |
| `plot_phi_lm`                 | Plots the (l,m) mode of one of Phi0, Phi1, or Phi2 as measured from a given detector.                                      |
| `plot_psi4_lm`                | Plots the (l,m) mode of Psi4 as measured from a given detector.                                                            |
| `plot_strain_lm`              | Plots the (l,m) mode of the gravitational-wave strain as measured from a given detector from Psi4.                         |
| `plot_total_luminosity`       | Plots the combined electromagnetic and gravitational-wave luminosity as measured from a given detector with Phi2 and Psi4. |

### Utilities

| Name                           | Description                                                                                                                                                                                          |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `print_ah_formation_time`      | Prints the time when the given apparent horizon was first found.                                                                                                                                     |
| `print_available_iterations`   | Given a variable, prints which iterations are available across all the data and all the dimensions. Optionally, you can specify only one dimension. This script is useful for writing shell scripts. |
| `print_available_timeseries`   | Prints all the various timeseries found in the data (reductions, scalars, ...). This script is useful for exploring the data available in a simulation.                                              |
| `print_grid_point`             | Prints where is the maximum or minimum (optionally, the absolute one) for a given grid function at a given iteration.                                                                                |
| `print_qlm_properties_at_time` | Prints some of the interesting properties (as from QuasiLocalMeasures) for a given horizon at a given time. Optionally estimates the Lorentz factor.                                                 |
| `save_reasampled_grid_data`    | Reads a grid function, resamples it to a given grid, and saves it to a file. This script is useful to move data from a cluster to your local machine (especially for 3D visualization)               |


## Movie files

`kuibit` plays nicely with
[motionpicture](https://github.com/Sbozzolo/kuibit/blob/master/examples/mopi_movies/grid_var),
a Python program to make animations. The following examples can be immediately
be used, for example, to make a movie of the logarithm of `rho_b` on the XY
plane:

``` sh
mopi -m grid_var --resolution 500 --plane xy --variable rho_b --colorbar --interpolation-method bicubic --vmin -7 --vmax -1 --parallel --outdir movie --logscale -x0 -30 -30 -x1 30 30
```

`motionpicture` requires `ffmpeg` to produce the final video. If it is not
available, `motionpicture` will only produce the frames. See

We recommend defining a `MOPI_MOVIES_DIR` environment variable and putting all
the files there. See
[motionpicture](https://sbozzolo.github.io/kuibit/motionpicture.html) for a
quick-start guide.

| Name       | Description                                |
|------------|--------------------------------------------|
| `grid_var` | Plot a given 2D variable on a given plane. |

