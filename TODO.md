# Wishlist

Here we collect ideas to improve and extend PostCactus. If you want to
contribute to the project, this a good place where to start. The projects are
sorted in no particular order. The number of [=] along with each idea indicates
the expected difficulty. This parameter increases with the level of knowledge of
PostCactus or of Python required to complete the task.

## Features

* Function to compute spectrogram of `TimeSeries`. [=]
* The extrapolation to infinity function for gravitational waves has to be tested
  and can be extended to support generic strains (not only for fixed l, m). [==]
* Improve algorithm for `__call__` in `Series` and `grid_data` to be more
  Pythonic and faster. [==]
* Extend `Series` and `grid_data` to support array data instead of only scalar
  data. [====]
* Correctly identify and merge refinement levels in `HierarchicalGridData` even
  where there are multiple centers of refinement. [===]
* Linear momentum lost by gravitational waves. [=]

* Add layer in class hierarchy in `cactus_grid_functions` to save the work done in
  reading ASCII files containing multiple variables. [==]
* Port `cactus_parfile` from `PostCactus2`. [==]
* Port `cactus_timertree` from `PostCactus2`. [==]
* Port support for grid data with reflection from `PostCactus2`. [==]
* Port support for locating bounding boxes from `PostCactus2`. [==]
* Add support for HDF5 for `AHFinderDirect` output. [==]
* Add method to merge `AHFinderDirect` patches. [==]
* Perform interpolation in shapes of `AHFinderDirect` to better find shapes when
  the cut is not on a major direction. [==]

* Add support for `all_reductions_in_one_file` to `cactus_scalars`. [==]

* Hunt for TODOs in the codebase and implement them. [=?=]

## Infrastructure

* Improve errors with dynamical name types. [==] (e.g. `type(self).__name__`
  instead of hardconding the name)
* Improve docstrings.  [==]
  (According to PEP8, the first line should be short and descriptive.)
  Check that all the sphinx param descriptions end with a period.
* Simplify `apply_unary`, `apply_binary`, `apply_to_self` in `numerical.py`.
  These functions may be turned into decorators, or at least used in a more
  concise way. [==]
* Numba-ify low-level functions. [====]
* `_multipoles_from_textfiles` takes a long time if there are thousands of
  files. [==]
* `cactus_scalars`, `cactus_multipoles`, `cactus_grid_functions` have a lot of
  common infrastructure. There should be a way to reduce code deduplication.
  Also, at the moment, the three group of classes are structured in very
  different ways, they should be more uniform. [====]
* Add support for `logging`. [===]
* Refactor code to correctly use `np.array` or `np.asarray` (not use `np.array`
  when is not needed). [==]
