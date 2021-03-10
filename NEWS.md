# Changelog

##  Version 1.0.0b1 (currently in development)

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
#### Bug fixes
- `cactus_horizons` and `cactus_multiploes` now remove duplicate iterations
#### Breaking changes
- `remove_duplicate_iters` was renamed to `remove_duplicated_iters`
