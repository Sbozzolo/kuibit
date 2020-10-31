# Wishlist

Here we collect ideas to improve and extend PostCactus. If you want to
contribute to the project, this a good place where to start. The projects are
sorted in no particular order. The number of [=] along with each idea indicates
the expected difficulty. This parameter increases with the level
of knowledge of PostCactus or of Python required to complete the task.

## Features

* Function to compute spectrogram of TimeSeries. [=]
* The extrapolation to infinity function for gravitational waves has to be tested
  and can be extended to support generic strains (not only for fixed l, m). [==]
* Improve algorithm for `__call__` in `Series` and `grid_data` to be more
  Pythonic and faster. [==]
* Extend `Series` and `grid_data` to support array data instead of only scalar
  data. [====]

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
  common infrastrucutre. There should be a way to reduce code deduplication.
  Also, at the moment, the three group of classes are structured in very
  different ways, they should be more uniform. [====]
* Uniform `keys()` function to Python3 standard. [==]
* Add support for `logging`. [===]
