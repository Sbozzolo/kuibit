# Changelog

## Version 1.1.0 (Under development)

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

#### General
- Added `bins` examples: `save_resampled_grid_data`

#### Features
- Added method to compute the linear momentum lost by gravitational waves
- Now the method `save` in `UniformGridData` supports `.npz` files. This is the
  recommend and fastest way to save a `UniformGridData` to disk.
- The function `load_UniformGridData` can now read `.npz` files.

#### Bug fixes
- Fixed bug that, under certain circumstances, resulted in `cactus_grid_function`
  not correctly indexing all the 3D data files
- Fixed a test that was triggering the wrong error.

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
