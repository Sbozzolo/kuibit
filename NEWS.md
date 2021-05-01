# Changelog

## Version 1.2.0 (Under development)

#### New module: visualize_matplotlib

The `visualize_matplotlib` module aims to simplify common visualization tasks
with `matplotlib`. At the moment, it mainly supports visualizing grid data and
apparent horizons outlines. The public functions in `visualize_matplotlib` try
to be as general as possible: if you pass some grid objects, they will try to
figure out how to plot it. Nonetheless, you should read the documentation and
the docstrings of the various functions.

#### General
- Improvements to documentation, docstrings, and tutorials

#### New examples

* `print_qlm_properties_at_time.py`

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
