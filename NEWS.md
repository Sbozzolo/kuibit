# Changelog

## Version 1.4.0 (2 May 2023)

#### General
- Minimum version of Python required is now 3.7. `kuibit` now explicitly follows
  [NEP29](https://numpy.org/neps/nep-0029-deprecation_policy.html).
- Started gradual typing
- Now `SimDir` can be imported directly from `kuibit` (`from kuibit import SimDir`)
- Update logo to high resolution (and vector graphics)

#### Features
- Added linear momentum lost by gravitational waves along the x and y directions
  (starting from a contribution by @konrad-topolski)
- Added `Tree` data structure
- Added `effective_amplitude_spectral_density` within `gw_utils` (starting from
  a contribution by @irrationalnumbers,
  [#27](https://github.com/Sbozzolo/kuibit/pull/27))
- Added fourth order accuracy in finite differences of grid functions
- Added `copy` to gravitational and electromagnetic waves
- Added `crop` and `cropped` to multipole data
- `setup_matplotlib` now takes an optional parameter `rc_param_file` to configure
  matplotlib using a file
- Added computation of linear momentum lost by electromagnetic waves
- Added `rotation180_symmetry_undo` to undo rotational symmetries in grid data
- Added `cactus_twopunctures` to read metadata from TwoPunctures
- Added `local_maxima` and `local_minima` to series
- Added `mean`, `average`, `median`, `std` reductions to series and grid data
- Added `ah_available`, `qlm_available` to `OneHorizon`
- Added `tensor` module
- Added support to `surface*.vtk` from `QuasiLocalMeasures`

#### Support to `VTK` variables from `QuasiLocalMeasures`

`QuasiLocalMeasures` has an option to output its variables on the horizon. Now,
`kuibit` can read and parse them, so it is possible to plot horizon meshes and
variables onto the mesh.

#### New `tensor` module

`kuibit.tensor` is a new module to provide high-level interfaces to work with
tensorial objects. A `Tensor` can be built with collections of `TimeSeries`,
`FrequencySeries`, `UniformGridData`, `HierarchicalGridData`, or any other class
derived from `BaseNumerical`. `Tensor`s support all the mathematical operations
and inherit the methods from their contained object. At the moment, there are
two new subclasses of `Tensor` that implement additional features for vector
calculus. For example, if `bh_cen` is a `Vector` with the time evolution of the
centroid of an horizon. `bh_cen.differentiated().norm()` will be a `TimeSeries`
with the magnitude of the coordinate velocity as a function of time (as a
`TimeSeries`).

Currently, the number of specialized methods in `Vector` and `Matrix` is not
large. Nonetheless, the infrastructure is in place and adding new will be
straightforward.

#### New `cactus_twopunctures` module

`TwoPunctures` creates a metadata file that stores important physical
information about the initial data. The new module `cactus_twopunctures` can
read this information and present it as a dictionary. If `sim` is a
`SimFactory`, `sim.twopunctures` will be a dictionary-like object that has as
keys the various quantities saved by `TwoPunctures` and as values, their values
as read from the `TwoPunctures.bbh` file.

#### New `tree` module

`kuibit.tree` is a new module that can represent tree structures. At the moment,
it is main use is to work with timer trees.

#### New `hor_utils` module

`kuibit.hor_utils` is a handy new module that implements common functions needed
when working with binary black holes. At the moment, the functions implemented
are:

- `compute_separation_vector`
- `compute_separation`
- `compute_center_of_mass`
- `compute_angular_velocity_vector`

#### Bug fixes

- Fix passing an empty `ah_vars` to `OneHorizon`.
- Fix floating-point comparison in `grid_data`.
- Fix optional positional arguments to `series._local_extrema`.
- Fix algorithm to compute hash for `UniformGrid`.
- Make classes that should not hashed unhashable.
- `GravitationalWaves` now find multipole variables that contain `psi4` (as
  opposed to being identically to `Psi4`)

#### Detector sensitivity curves

- Add more updated sensitivity curve for Einstein Telescope (ET-D) (thanks,
  @bgiacoma, [#33](https://github.com/Sbozzolo/kuibit/issues/33)).

#### New/updated examples

Scripts:

* `describe_simdir.py`
* `interactive_timertree.py`
* `print_available_iteration.py` optionally prints the corresponding time
* `plot_charge_phi1.py`
* `plot_binary_ah_angular_velocity.py`
* `plot_binary_ah_period.py`


## Version 1.3.6 (5 July 2022)

#### Bug fixes
- Fix time incorrectly read in ASCII data (thanks @AuroraDysis,
  [#30](https://github.com/Sbozzolo/kuibit/issues/30))

## Version 1.3.5 (14 June 2022)

#### Bug fixes
- Fix str representation of multipoles
- Fix implicit plotting of `Series` for `matplotlib >= 3.5.2`
- Fix `test_init` in `test_cactus_grid_function`

## Version 1.3.4 (22 April 2022)

#### General
- Improvements to documentation and docstrings

#### Bug fixes
- Added error in `_plot_horizon_on_plane` when the horizon cannot be plotted
- Updated NumPy types (np.int -> int, np.float -> float)
- Windowing uneven signals is no longer allowed
- Correctly remove ghost zones for 1-2D HDF5 files
- Fix deprecation warning for Matplotlib 3.4
- Fix `is_masked` after removing a mask (thanks @ekwessel,
  [#28](https://github.com/Sbozzolo/kuibit/issues/28))
- Fix some information not being propagated by
  `grid_data_utils.merge_uniform_grids`.

#### Breaking changes
- The `merge_refinement_levels` function in `HierarchicalGridData` has been
  renamed to `refinement_levels_merged` for consistency. (Imperative methods
  edit the object in-place)

#### New examples

Scripts:

* `plot_ah_trajectories.py`

## Version 1.3.3 (19 November 2021)

#### Bug fixes

- Fixed `tikzplotlib` dependency for `Python 3.6`

## Version 1.3.2 (16 November 2021)

#### Bug fixes

- Examples with `plot_components_boundaries` now respect axes limits
- Improved support to installing `mayavi`
- `Python 3.10` is now supported (with exception of `numba`)
- Fixed parity in `reflection_symmetry_undo`

## Version 1.3.1 (4 November 2021)

#### Bug fixes

- `h5py >= 3` is now supported

## Version 1.3.0 (28 October 2021)

#### `SimDir` can now be cached in pickle files

`kuibit` tries to do as much lazy-loading as possible. For examples, files are
opened only when needed. When analyzing simulations it is useful to save the
work done by `kuibit` to avoid re-doing the same operations over and over. It is
now possible to do this using pickle files. `SimDir` can now be used as a
context manager and the progresses can be loaded and saved from files. For
example:

```python
with SimDir("path_of_simulation", pickle_file="simdir.pickle") as sim:
    # do operations
```

In this case, if `pickle_file` exists, it will be loaded (ignoring all the other
arguments passed to `SimDir`), and it will be kept updated with the additional
work done by `kuibit`. If `pickle_file` does not exist, the `SimDir` will be
created as usual as a `pickle_file` will be generated.

It is important to stress that, when using pickles, no consistency check with
the current state of the simulation is performed. If the simulation changes
(e.g., new checkpoints are added), this will result in errors. In that case, a
new pickle file must be produced.

#### Masked data

Numerical objects now support
[mask](https://numpy.org/doc/stable/reference/maskedarray.generic.html), which
can be used to ignore part of the data that satisfy certain conditions (for
example, to exclude the atmosphere from GRMHD computations).

Note that it is not possible to perform interpolation with masked data, so
several methods will not work.

Note also that we mask only the data, not the independent coordinate (e.g., the
time in TimeSeries or the spatial coordinates in the UniformGridData). In case
you need masked coordinates too, you can use the `mask` method to obtain an
array of booleans that identifies the valid data.

- `Series`, `UniformGridData`, `HierarhicalGridData` have a new method
  `is_masked`.
- `Series`, `UniformGridData`, `HierarhicalGridData` have a new method
  `mask` that identifies where data is invalid.
- `Series`, `UniformGridData`, `HierarhicalGridData` have news methods to create
  masked data (e.g., `mask_greater`). See complete list in documentation.
- `Series` have new methods `mask_remove` and `mask_removed` to create objects
   without masked data.
- `Series` have new methods `mask_apply` and `mask_applyed` to create objects
   with a given mask (as the one obtained with the `mask` method).

#### General
- `SimDir` can be saved to disk with the method `save` and read with the
   function `load_SimDir`. This is useful to work with a simulation that has
   finished.
- Examples can now use pickles.
- New CI workflow: linting.
- New `First Steps` documentation page.
- When a release is published, its documentation is saved to
  `sbozzolo.github.io/kuibit/VERSION`

#### Features

- `time_at_maximum` and `time_at_minimum` in `TimeSeries` can now take the
  optional argument `absolute`.
- Added `x_at_minimum_y` and `x_at_maximum_y` to `BaseNumerical`.
- Added `coordinates_at_maximum` and `coordinates_at_minimum` for grid data.
- Added `HierarchicalGridData.is_complex()`.
- Added `tikz_clean_figure` to `visualize_matplotlib.save`, to
  `argparse_helper.add_figure`, and to examples. This can be used to reduce
  the size of output `tikz` files.
- Added `clear_cache` in `OneGridFunction`.
- Added `plot_contour`.
- Added alias `time_found` to `formation_time` in horizons.
- Added `plot_components_boundaries`.
- Added `ghost_zones_remove` in `HierarchicalGridData`
- Added `add_grid_structure_to_parser`.
- Added `reflection_symmetry_undone`.

#### Breaking changes
- The `ignore` parameter in `SimDir` has been renamed to `ignored_dirs`.
- The `trim_ends` parameter in `cactus_waves` is now set to `False` by default.

#### Bug fixes

- `plot_colorbar` does not steal axis focus anymore.
- The legend in `plot_psi4_lm` was corrected.
- `visualize_matplotlib.save` now correctly supports the `figure` argument.
- `plot_strain_lm.py` no longer crashes when `window_args` is not provided.
- `HierarchicalGridData` now owns the components.
- Clear `OneGridFunction` cache in `grid_var` to avoid death by OOM.
- Uniform constructor of `GridSeries` with constructors of other `Series`.
- Horizon properties now lead to valid python variable names.

#### New examples

Examples with `--detector-num` now also accept `--num-detector` as alias.

Scripts:

* `picklify.py`
* `plot_1d_slice.py`
* `plot_grid_expr.py`
* `plot_phi_time_averaged.py`
* `plot_gw_angular_momentum.py`
* `print_grid_point_minmax.py`

## Version 1.2.0 (1 June 2021)

#### New module: visualize_matplotlib

The `visualize_matplotlib` module aims to simplify common visualization tasks
with `matplotlib`. At the moment, it mainly supports visualizing grid data and
apparent horizons outlines. The public functions in `visualize_matplotlib` try
to be as general as possible: if you pass some grid objects, they will try to
figure out how to plot it. Nonetheless, you should read the documentation and
the docstrings of the various functions.

#### `motionpicture`

Making movies is a critical step in analyzing a simulation. Now, `kuibit` comes
with [motionpicture](https://github.com/Sbozzolo/motionpicture), a Python tool
to assist you animate your data. `motionpicture` provides all the infrastructure
needed to render multiple frames and glue them together, so, all you need to
worry is how to render one single frame. Importantly, `motionpicture` supports
parallel rendering, which can dramatically speed up the time needed to produce a
video. Check out the
[grid_var](https://github.com/Sbozzolo/kuibit/blob/master/examples/mopi_movies/grid_var)
example to see how easy it is to make a movie.

#### New class: GridSeries

`GridSeries` is a new class to describe 1D grid data. This class utilizes the
same infrastructure used by `TimeSeries` and `FrequencySeries` to represent a
single-valued function. `UniformGridData` can be transformed into `GridSeries`
with the method `to_GridSeries`. The main reason you would want to do this is
because `GridSeries` are leaner and more direct to use.

#### General
- Improvements to documentation, docstrings, and tutorials
- Examples are now automatically packaged and uploaded upon release
- New YouTube series, `Using kuibit`

#### New examples

Scripts:

* `plot_1d_vars.py`
* `plot_ah_coordinate_velocity.py`
* `plot_ah_found.py`
* `plot_ah_radius.py`
* `plot_ah_separation.py`
* `plot_constraints.py`
* `plot_em_energy.py`
* `plot_grid_var.py`
* `plot_gw_energy.py`
* `plot_gw_linear_momentum.py`
* `plot_physical_time_per_hour.py`
* `plot_phi_lm.py`
* `plot_psi4_lm.py`
* `plot_strain_lm.py`
* `plot_timeseries.py`
* `plot_total_luminosity.py`
* `print_qlm_properties_at_time.py`

Movies:

* `grid_var`

#### Bug fixes
- Fixed header recognition for `carpet-grid.asc` (#22)

#### Features

- New methods `get_apparent_horizon` and `get_qlm_horizon` in `HorizonsDir`.

## Version 1.1.1 (22 April 2021)

#### Bug fixes
- Fixed corner in `__str__` in `UniformGrid`
- Fixed `_finest_component_at_point_mapping` for some points for which floor and
  rounding lead to different results.

## Version 1.1.0 (18 April 2021)

#### New module: argparse_helper

The `argparse_helper` module collects functions to set up arguments for
command-line scripts. It comes with options for working with figure, grid data,
and horizons. Options can be stored in text files and read passing the path to
the `-c` flag.

#### Faster `HierarchicalGridData`

The function that finds the component corresponding to a given point in
`HierarchicalGridData` was significantly sped up by adopting the algorithm used
in `PostCactus` (developed by Wolfgang Kastaun). This new algorithm is
significantly faster, but it will not work in case the refinement factors across
different levels are not constant integers. In that case (e.g., the refinement
boundaries are at 1, 2, and 6), the older algorithm will be used. For large
towers of refinement levels and hundreds of MPI processes, the new algorithm is
orders of magnitude faster.

#### Examples

Now `kuibit` comes with runnable examples. These are production-grade codes that
you can immediately use for your simulations. They are a great way to learn about
how to use `kuibit`. The examples included are:

* `print_ah_formation_time.py`
* `print_available_iterations.py`
* `print_available_timeseries.py`
* `save_resampled_grid_data.py`

#### General

- Releases are now automatically pushed to PyPI.

#### Features
- Added method to compute the linear momentum lost by gravitational waves
- Now the method `save` in `UniformGridData` supports `.npz` files. This is the
  recommend and fastest way to save a `UniformGridData` to disk.
- The function `load_UniformGridData` can now read `.npz` files.

#### Bug fixes
- Fixed bug that, under certain circumstances, resulted in `cactus_grid_function`
  not correctly indexing all the 3D data files
- Fixed a test that was triggering the wrong error.
- Fixed a bug that made `ra_dec_to_theta_phi` depend on the local time

#### Breaking changes
- `finest_level_component_at_point` is now `finest_component_at_point` and
   returns directly the component as `UniformGridData` as opposed to the
   refinement level and component number.

## Version 1.0.0 (11 April 2021)

#### General
- Improvements to documentation and docstrings

#### Features
- Added `nanmax`, `nanmix`, `abs_nanmax`, `abs_nanmin` to `BaseNumerical`
- Added support for HDF5 grid arrays
- Added slicing `HierarchicalGridData`
- Added `shape_time_at_iteration` in `cactus_horizon`
- Added `shape_at_time` in `cactus_horizon`
- Added `shape_outline_at_time` in `cactus_horizon`
- Added `compute_horizons_separation` in `cactus_horizon`
- Added `ignore_symlinks` to `SimDir`

#### Bug fixes
- `cactus_horizons` and `cactus_multiploes` now remove duplicate iterations

#### Breaking changes
- `remove_duplicate_iters` was renamed to `remove_duplicated_iters`
- `_derive` methods are renamed to`_differentiate`

## Version 1.0.0b0 (11 January 2021)

Initial release
