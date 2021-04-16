# Changelog

## Version 1.1.0 (Under development)

#### New module: argparse_helper

The `argparse_helper` module collects functions to set up arguments for
command-line scripts. It comes with options for working with figure, grid data,
and horizons. Options can be stored in text files and read passing the path to
the `-c` flag.

#### Bug fixes
- Fix bug that, under certain circumstances, resulted in `cactus_grid_function`
  not correctly indexing all the 3D data files

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
