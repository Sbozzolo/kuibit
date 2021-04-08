List of features implemented in kuibit
==================================================

``kuibit`` implements a large collection of features. In general, the spirit
of the package is to provide high-level interfaces to simulation data, hiding
all the computation complexity behind it. In doing this, ``kuibit`` tries to
be Pythonic: many custom objects behave either like dictionaries, or like
callable. These objects behave as you expect (or at least, they should). For
example, if an objects smells like a dictionary, you should be able to ask for
``keys()``, or iteratate over it, or see if an element is contained with the
``in`` keyword.

Here we review all the available features as of version ``3.0.0b1``.

General
-------

- ``kuibit`` is documented, thoroughly commented, provides examples, and has
  a large test suite.
- ``kuibit`` is available as wheel and can be installed with ``pip``.

SimDir
------

- Scan and organize all the files in a directory up to a specified depth.
- Print out an overview of what is available in the simulation.
- Individuate all the par, log, out, and err files.
- Easily access all the data that ``kuibit`` can represent.
- Transparently handle all the restarts with any directory structure.

Series
------

Time and frequency series are represented in an intuitive way. They

- can be real or complex;
- can be unevenly spaced;
- support all the mathematical operations (e.g., you can sum two timeseries);
- can be interpolated with arbitrary (up to quintic) splines;
- can be called returning interpolated values where no data is available
- support reductions (maximum/minimum, absolute maximum/minimum, location of the maximum/minimum, location of the absolute maximum/minimum);
- are compatible with NumPy's operations (e.g., you can all ``np.log(rho)``);
- are compatible with matplotlib ``plot`` (you can plot with ``plt.plot(rho)``);
- can be resampled using nearest neighors or splines on new times/frequencies;
- can be written to disk in a human-readable way (with ``save``);
- can be clean from ``nans``;
- can be integrated with the trapezional method (cumulative integral);
- can be derived from splines, or with second-order finite differencing;
- can be smoothed with the Savitzky-Golay filter;
- can be cropped;
- can be resampled to the points common points with other series.

Specifically timeseries, also support:

- computing the unfolded phase and phase velocity;
- computing duration, time intervals;
- shifting time/phase, aligning at (absolute) maximum/minimum;
- resampling evenly, even with a fixed frequency or delta time;
- zero-paddeding;
- removing mean;
- removing initial/final portion of the signal;
- changing time units;
- redshifting for a given redshift;
- window functions (Tukey, Hamming, Blackman, or arbitrary);
- Fourier Transform (real and complex).

Frequencyseries support:

- loading from file (e.g., noise curves);
- normalization;
- low/high pass filters;
- removing negative frequencies;
- locating peaks (with also a quadratic approximation);
- Inverse Fourier Transform;
- computing the inner product between two frequencyseries, also with multiple noises;
- computing the overlap, also with multiple noises.

Scalar data
-----------

- Read data from ``CarpetASCII`` (max, min, norm, ...) as timeseries (both ``one_file_per_group`` or ``one_variable_per_group`` options).
- Combine all the different files from multiple checkpoints in a single timeseries.
- Read data transparently compressed with gzip or bzip2.

Multipoles and waves
--------------------

- Read ASCII and HDF5 data from ``Multipoles``.
- Represent multipoles with objects that can be accessed with the multipole numbers.
- Use the fixed frequency integration method to compute from ``Psi4``:

  - strains at given multipole number,
  - strains at a given point (accounting for the spin-weigthed spherical harmonics),
  - strain as observed by LIGO/Virgo (considering the antenna patterns),
  - power/energy lost via gravitational waves (one or multiple modes),
  - torque/angular momentum along the z axis lost via gravitational waves (one or multiple modes),
  - Compute the last two for electromagnetic waves from ``Phi2``.

- Extrapolate waves at infinity with polynomial expansion in real/imaginary parts or amplitude and phase.
- Compute spin-weigthed spherical harmonics.
- Convert from RA and Dec to spherical coordiantes.
- Compute antenna responses.
- Compute signal to noise ratio from strain.
- Compute redshift from luminosity distance.
- Compute mismatch between the 2,2 modes of two waves for multiple detectors.
- Access sensitivity curves of known detectors (e.g., LISA, or Cosmic Explorer).

Units
-----

- Convert from geometrized units (given mass, length, or mass in solar masses) to physical and vice versa.
- Implement some basic constants of Nature.

Grid Data
---------

- Read 1D, 2D, and 3D ASCII and HDF5 files as ``HierarchicalGridData``, which supports:

  - working with multiple components and refinement levels;
  - handling ghost-zones;
  - merging multiple patches that logically represent a single grid (e.g., due to domain decomposition);
  - real or complex data;
  - all the mathematical operations (e.g., you can sum two timeseries);
  - interpolation with multilinear interpolation;
  - being called returning interpolated values where no data is available;
  - reductions (maximum/minimum, absolute maximum/minimum, location of the maximum/minimum, location of the absolute maximum/minimum);
  - NumPy's operations (e.g., you can all ``np.log(rho)``);
  - resampling using nearest neighors or splines on new grids;
  - Second-order finite-differencing along any dimension;
  - being resampled to ``UniformGridData`` (unigrid);
  - abitrarily slicing with lower-dimensional cuts (e.g., equatorial plane from 3D data).

- In addition to above ``UniformGridData`` support:

  - being saved on disk;
  - histogram and percentiles;
  - additonal reductions (e.g., norm2, mean, norm-p, integral);
  - changing grid spacing (up/down sampling);
  - Fourier Transform;
  - computing grid coordiantes (for plotting or operations involving the coordinates);

- Read multiple iterations as spacetime ``HierarchicalGridData`` (to take advantage of multilinear interpolation in space and time).
- Transparently handle multiple restarts/output from different MPI processes.
- Computing the total size of the files associated to a variable/dimension.

Horizons
---------

- Read and represent the ASCII output from ``QuasiLocalMeasures`` and ``AHFinderDirect``.
- Work with the shape of the horizons and their properties (as timeseries).
- Cut the 3D shape into 2D projection along the axes centered in the origin of the horizon.
