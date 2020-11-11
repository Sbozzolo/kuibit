# News

## 3.0.0

With version 3.0.0, `PostCactus` has been completely rewritten. Here is a
summary of the changes:

### General

* Support to versions of Python prior to 3.6 is dropped.
* Documentation and examples are now available.
* `unittest` is now being used for tests.
* `PostCactus` now adheres to PEP8.
* `PostCactus` now uses a descriptive naming convention for variables and
  functions.
* `PostCactus` now uses modern a distribution and build Python infrastructure
  based on `pyproject.toml` as opposed to `setup.py`.

### Time and Frequency Series

* A new abstract class, `BaseSeries`, is available. `BaseSeries` represent any
  kind of function `y(x)` in which `x` is monotonically increasing. `BaseSeries`
  is based on another abstract class `BaseNumerical`, which implements a number
  of useful methods to handle such data, including all the mathematical
  functions. This is shared with grid functions.
* The `TimeSeries` class is now derived from `Series` and is enriched with new
  methods.
* A new class `FrequencySeries`, derived from `Series`, was introduced to
  represent frequency series. The new class comes with several methods, like
  `overlap`.

### Grid functions

* Grid-related classes have been renamed and are now derived from
  `BaseNumerical`. Much of the inner workings have been rewritten.
* The readers are completely redesigned and are now much simpler to understand
  and maintain.
* `PostCactus` can now read ASCII data in arbitrary dimension.

### Multipoles

* Multipoles are now represented in a way that is more consistent with the other
  objects and the implementation is now simplified.

### Gravitational Waves

* Dropped support for gravitational waves extracted with the Zerilli-Moncrief
  method.
* Added functions to compute mismatch between two gravitational waves.
* Added functions to compute antenna patterns of known interferometers and to
  compute spin-weighted spherical harmonics.
* Added sensitivity curves for known detectors.

### Electromagnetic Waves

* Added support for electromagnetic waves as extracted from `Phi2`.

### Horizons

* Dropped support for IsolatedHorizons.
* Simplified horizon-related classes and merged into `OneHorizon`.
