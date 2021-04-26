# Examples of working problems built with `kuibit`

In this directory, we collect working programs built with `kuibit`. You can use
these scripts as good examples of ``kuibit`` usage (or you can directly use
them).

> :warning: While `kuibit` is tested at each commit to ensure that nothing
>           breaks, these codes are not. If you find one that does not work,
>           please report it and we will fix that.

## Scripts available

### Utilities

| Name                           | Description                                                                                                                                                                                          | Last tested with |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| `save_reasampled_grid_data`    | Reads a grid function, resamples it to a given grid, and saves it to a file. This script is useful to move data from a cluster to your local machine (especially for 3D visualization)               | `1.1.0`          |
| `print_ah_formation_time`      | Prints the time when the given apparent horizon was first found.                                                                                                                                     | `1.1.0`          |
| `print_available_iterations`   | Given a variable, prints which iterations are available across all the data and all the dimensions. Optionally, you can specify only one dimension. This script is useful for writing shell scripts. | `1.1.0`          |
| `print_available_timeseries`   | Prints all the various timeseries found in the data (reductions, scalars, ...). This script is useful for exploring the data available in a simulation.                                              | `1.1.0`          |
| `print_qlm_properties_at_time` | Prints some of the interesting properties (as from QuasiLocalMeasures) for a given horizon at a given time. Optionally estimates the Lorentz factor.                                                 | `1.1.0`          |
