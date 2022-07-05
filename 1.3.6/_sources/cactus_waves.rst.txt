Gravitational waves
==============================

``kuibit`` has a large number of features to work with gravitational (and
electromagnetic) waves as extracted with the Newman-Penrose formalism. Most of
these features are inherited from other objects: multipoles and timeseries, as
gravitational waves are typically studied in a multipolar decomposition and they
are functions evolving with time. For this reason, you should familiarize
yourself with the objects in ``kuibit`` that represent those quantities:
:ref:`series:Time and frequency series` and
:ref:`cactus_multipoles:Working with multipolar decompositions`.

The main two objects to work with gravitational waves are
:py:class:`~.GravitationalWavesOneDet` and :py:class:`~.GravitationalWavesDir`.
The first described waves as seen by one specific detector (fixed radius), the
second collects :py:class:`~.GravitationalWavesOneDet` for all the available
detectors. These classes inherit several features from the base version in the
:py:mod:`~.cactus_multipoles` module (:ref:`cactus_waves_ref:Reference on kuibit.cactus_waves`).

.. note::

   While this page is devoted to gravitational waves, electromagnetic waves are
   implemented in a similar way. It is important to understand what kind of
   electromagnetic waves we are considering here: those that are extracted with
   the Newman-Penrose formalism. Most people will not deal with this kind of
   electromagnetic waves. An example of electromagnetic waves of this kind can
   be extracted with the ``Proca`` thorns.

Accessing gravitational-wave data
---------------------------------

Most of the interesting functions related to gravitational waves (with exception
of the extrapolation to infinity) are for a fixed extraction radius. Hence, they
live in the class :py:class:`~.GravitationalWavesOneDet`. This object contains
all the available multipoles for ``Psi4`` at a fixed radius.

A typical workflow starts from a :py:class:`~.SimDir`:

.. code-block:: python

    sd = SimDir("gw150914")
    gws = sd.gws
    gw_r100 = gws[100]  # This is a GravitationalWavesOneDet

``gw_r100`` contains all the information on ``Psi4`` at radius 100. There are
multiple ways to get the actual timeseries for a specific monopole. For instance
``psi4_22 = gw_r100[(2, 2)]``.

The more interesting quantity is the gravitational-wave strain. To access this
for a specific multipole, use the method :py:meth:`~.get_strain_lm()`.

Strain
______

The method :py:meth:`~.get_strain_lm()` return the strain for the multipolar
component (l, m) multiplied to the extraction distance. This is done via
double-time integration using the method of the fixed-frequency in Fourier
space. The fixed-frequency integration reduces drifts in the mean value of the
strain.

This function requires some care to be produce meaningful results.

First, to avoid aliasing and spectral leakage, the timeseries must go to zero at
the boundaries of the interval. If the physical signal does not have this
property, a window function must be applied. This can be done directly with
:py:meth:`~.get_strain_lm()` providing the argument ``window_function``.

If ``window_function`` is ``None``, no window is applied. Alternatively, one can
apply one the windows already defined in the :py:mod:`~.timeseries` module. To
do this, just pass a string with the name of the window. You can find these
names looking at the methods in :py:class:`~.TimeSeries` and finding those that
end with ``windowed``: the first part of the name is what you have to pass
(e.g., ``tukey``). Alternatively, you can pass a function that takes as first
argument the length of the data and returns an array with the window (this is
how windows in SciPy are implemented). In both cases, if the window requires
additional parameters, you can pass them providing them directly to
:py:meth:`~.get_strain_lm()`.

Second, you must provide a ``pcut`` parameter. This is required by the
fixed-frequency integration method. ``pcut`` is typically chosen as the longest
physical period in the signal (or the shortest frequency). In the case of a
binary inspiral, this is approximately the period of the first half orbit. The
fixed-frequency integration suppresses smaller frequencies signals.

Finally, because of the windowing and the integration, signals around the
boundaries is not too reliable. It is removed by default. You can opt-out
setting ``trim_ends`` to ``False``.


In case you are interested in summing up all multipole monopoles, you should use
the more general function :py:meth:`~.get_strain()`. This function takes input
similar to :py:meth:`~.get_strain_lm()`, and requires to specify an evaluation
angle ``(theta, phi)``. In case you want to sum up only up to a given :math:`l`,
pass the argument ``l_max``.

Similarly, you can compute what would be the gravitational wave strain observed
by the LIGO-Virgo interferometers using :py:meth:`~.get_observed_strain` and
providing a sky localization. This method computes the strain and convolves it
with the antenna responses :math:`F` of the single detectors:

.. math::

   h = F_\times h_\times(\theta_{\mathrm{GW}}, \phi_{\mathrm{GW}}) + F_+ h_+(\theta_{\mathrm{GW}}, \phi_{\mathrm{GW}})

Here, :math:`\theta_{\mathrm{GW}}` and :math:`\phi_{\mathrm{GW}}` are the
spherical coordinates of the observer from the binary's frame, taking the
angular momentum of the binary to point along the z-axis. This function does not
add noise.

Extrapolate_to_infinity
^^^^^^^^^^^^^^^^^^^^^^^

The function :py:meth:`~.extrapolate_strain_lm_to_infinity` can be used to
extrapolate gravitational waves strain to spatial infinity. This is done fitting
polynomials to wavefronts that are aligned in retarded times (assuming a
background Schwarzschild spacetime). The most important paramter that this
function takes is the list of distances that you want to use for the
extrapolation. These have to be distances at which you have detectors. You can
find all the available distances with the ``keys()`` method.

For improved stability, you can extrapolate the waves as amplitude and phase
(instead of real and imaginary parts). To do this, provide the
``extrapolate_amplitude_phase=True`` option.

TODO

   Expand this section.

.. warning::

   This function has not been thorougly tested!


Energy and angular momentum
___________________________


:py:class:`~.GravitationalWavesOneDet` (and
:py:class:`~.ElectromagneticWavesOneDet`) implements methods to compute the
instantaneous power and torque along the z axis. To compute these quantities for
a specific mode, just use :py:meth:`~.GravitationalWavesOneDet.get_power_lm` or
:py:meth:`~.get_torque_z_lm`. You can also compute these quantities for all
the available multipoles up to a given ``l_max`` using the methods
:py:meth:`~.GravitationalWavesOneDet.get_total_power` or :py:meth:`~.get_total_torque.` The
integrated (cumulative) versions are also available. These are the energy and
angular momentum in gravitational waves. The methods have similar names with
``energy`` instead of ``power`` and ``angular_momentum`` instead of ``torque_z``.

..
   .. note::

      Every time a function returns a tuple with the two gravitational-wave
      polarizations, the order is always alphabetical: the first element is the
      cross polarization, the second is the plus. However, in most cases, the
      preferred output is a complex number (or array of numbers).
