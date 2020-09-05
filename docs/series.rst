Time and frequency series
==============================

Time and frequency series are fundamental objects in most simulations.
``PostCactus`` supports them through the ``timeseries`` and ``frequencyseries``
modules, which define the ``TimeSeries`` (:ref:`timeseries_ref:Reference on
postcactus.timeseries`) and ``FrequencySeries``
(:ref:`frequencyseries_ref:Reference on postcactus.frequencyseries`) object
types. Both the classes are derived from a base class ``BaseSeries`
(:ref:`series_ref:Reference on postcactus.series`) which defines common methods
and features, so ``TimeSeries`` and ``FrequencySeries`` share a lot of
functionalities.

The TimeSeries and FrequencySeries objects
------------------------------------------

Here we describe some common feature for  ``TimeSeries`` and
``FrequencySeries``, and we leave for later sections to go into detail about
either. For clarity, we specifically talk about ``TimeSeries``, but everything
we discuss here applies to ``FrequencySeries`` as well.

Defining time series is easy:

.. code-block:: python

    import postcactus.timeseries as ts
    # or import postcactus.frequencyseries as fs
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 100)
    sin_wave = ts.TimeSeries(t, np.sin(t))

The object ``sin_wave`` has a number of useful features.

You can add, multiply, and divide two ``TimeSeries`` (``FrequencySeries``) with
the same times (frequencies) and scalars. For instance

.. code-block:: python

    cos_wave = ts.TimeSeries(t, np.cos(t))
    tan_wave = sin_wave / cos_wave

The operations currently implemented are sum, subtraction, multiplication,
division, power, absolute value, negative.

``TimeSeries`` and ``FrequencySeries`` objects are extremely powerful: they try
to behave like ``numpy`` arrays. For example, many of the mathematical functions
from ``numpy`` can be applied directly:

.. code-block:: python

    ones = np.arctan(tan_wave)
    complicated_expr = np.tanh(sin_wave ** 2 / np.abs(cos_wave))

Both the Series are callable objects, so they behave like normal functions. If
``t0`` is any arbitrary time (or frequency), you can call ``sin_wave(t0)`` to
get the value of ``sin_wave`` at `t0`. To do this, ``baseSeries`` uses spline
representation. For more information, read the documentation on splines.

``TimeSeries`` and ``FrequencySeries`` objects have multiple useful functions.
We follow this convention: methods with an name that is an imperative (e.g.,
``zero_pad``) modify the object does not return anything, methods with name that
is a past-tense verb (e.g., ``zero_padded``) return a new object with the
modification applied.

splines
^^^^^^^^^^^^^^^^^^^^^^^^

One of the most powerful features of ``TimeSeries`` (``FrequencySeries``) is
that they are callable objects and they can evaluate the data at any arbitrary
time. This is done using splines. When you initialize a ``TimeSeries``
(``FrequencySeries``) a cubic spline representation with no smoothing (the
spline evaluates exactly to the data) is computed. This is cached into the
attributes ``self.spline_real`` (and ``self.spline_imag`` if the data is
complex).

Every time you modify the series (e.g., ``integrate``), the spline is updated.

This representation allows you to call the ``Series`` directly, but if you do it
outside the range of definition, you will get a ``ValueError``. You can change
the behavior of how to treat external data by calling directly
``evaluate_with_spline``. This takes a keyword argument ``ext``. Values outside
the interval are extrapolated if ``ext=0``, set to 0 if ``ext=1``, a ValueError
is raised if ``ext=2``, or if ``ext=3``, the boundary value is returned.

.. warning::

   Splines are good continuous representation of data, but they are not perfect,
   and they are especially unfit for discontinuous data. Be sure to understand
   the limitations, and use splines only when you know that the representation
   is good.

integrate
^^^^^^^^^

Integrate the ``TimeSeries`` (``FrequencySeries``) as a cumulative sum weighted
on the time intervals (trapezoid). The result is a new ``Series`` with the
integral as a function of time.

derive, spline_derive
^^^^^^^^^^^^^^^^^^^^^

The method ``derive`` derives ``Series(order)`` with a centered difference
method in the interior and a one-sided difference on the boundary. The operation
is applied ``order`` times to obtain a high-order derivative. On the other hand,
``spline_derive`` uses the spline representation to achieve the same task.
``spline_derive`` is typically and better behaved for nice enough timeseries.
You should not trust the values at the boundaries too much, you may want to crop
it out.

save
^^^^

Save the ``Series`` as an ASCII file with 2 columns :math:`(t, y)` for real
valued data and 3 columns :math:`(t, \Re (y), \Im (y))` for complex-valued ones.
The back-end is ``np.savetxt``, so you can provide additional arguments, like an
header.

savgol_smooth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``savgol_smooth(window_size, order)`` smooths the series with a Savitzky-Golay
filter with window of size ``window_size`` and order ``order``. This is just
like a regular "Moving average" filter, but instead of just calculating the
average, a polynomial (usually 2nd or 4th order) fit is made for every point,
and only the "middle" point is chosen. Since 2nd (or 4th) order information is
concerned at every point, the bias introduced in "moving average" approach at
local maxima or minima, is circumvented. At the moment, this is the preferred
way to smooth series.

iter
^^^^

Series are iterable, so you can do

.. code-block:: python

   for t, y in timeseries:
       print(t, y)


The TimeSeries methods
-----------------------

mean_remove, nans_remove
^^^^^^^^^^^^^^^^^^^^^^^^

``mean_remove``, as the name suggests removes the mean value from the
``TimeSeries``. Similarly, ``nans_remove`` filters out those data points with
infinitive or NaN values. The resulting ``TimeSeries`` has different number of
points.


time_unit_change, redshift
^^^^^^^^^^^^^^^^^^^^^^^^^^

``time_unit_change(T, inverse=False)`` rescales the time so that what was
previously ``T`` units of time now are 1. For example, if initially the units
where seconds, with ``T=1e-3`` the new units will be milliseconds. The keyword
argument ``inverse`` changes the direction: when ``inverse=True``, 1 unit of old
time becomes ``T`` units in the new time. This is useful to move from
computational units to physical units using the ``unitconv`` module.

The method ``redshift(z)`` uses ``time_unit_change`` to redshift the data by a
factor of :math:`1+z`.

.. code-block:: python

    import postcactus.unitconv as uc

    # Gravitational waves in geometrized units
    gw_cu = TimeSeries(...)

    # Gravitational waves in seconds, assuming a mass of 1 M_sun
    CU = uc.geom_umass_msun(1)
    gw_s = gw_cu.time_unit_changed(CU.time, inverse=True)


resample, regular_resample, fixed_frequency_resample, fixed_timestep_resample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``resample`` is a generic method to use splines to resample the ``TimeSeries``
to new times. Typical use-cases of ``resample`` have their of methods:
``regular_resample`` resamples to linearly space times,
``fixed_frequency_resample`` and ``fixed_frequency_resample`` resample the
timeseries with a provided timestep or frequency starting at ``tmin`` and ending
at a ``tmax`` that is an integer multiple of the timestep (or reciprocal of the
frequency).

Before using these methods, read the warning in ``make_spline``!

Fourier transform (to_FrequencySeries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can compute the discrete Fourier transform of a ``TimeSeries`` with the
``to_FrequencySeries`` method. This uses NumPy's ``fft`` module, so the
conventions are the same. The zero frequency is at the center of the array.

.. note::

   You are responsible of pre-processing the data (removing mean, windowing,
   etc.)


unfolded_phase, phase_angular_velocity, phase_frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``unfolded_phase`` returns a new ``TimeSeries`` with the (complex) unfolded
phase of the signal. If the signal is real, the unfolded phase is zero.
``phase_angular_velocity`` returns the derivative of the ``unfolded_phase``. The
derivative can be compute with finite difference by setting
``use_splines=False``, otherwise it is computed with the splines. Optionally,
the output can be smoothed over timescales of ``tsmooth`` with the
``savgol_smooth_time`` method. In this case, the ``TimeSeries`` is resampled to
regular timesteps. ``phase_frequency`` is just ``phase_angular_velocity``
divided by :math:`2\pi`, which is the angular frequency of the phase.

savgol_smooth_time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often, one knows the smoothing length in units of time as opposed to number of
points (e.g., I want to smooth over timescales of one second).
``savgol_smooth_time`` takes smoothing timescale as opposed to the window size.
To ensure consistency, ``savgol_smooth_time`` resamples the timescale to uniform
timesteps. When you have a regularly sampled timeseries, this function is more
direct than ``savgol_smooth``. However, when the sampling is very irregular in
time, the smoothing length changes throughout the timeseries (which is probably
something you do not want).

windowed, tukey_windowed, hamming_window, blackman_window
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``window(window_function)`` applies window_function to the timeseries.
``window_function`` has to be a function that takes as first argument the number
of points of the signal. ``window_function`` can take additional arguments as
passed by ``windowed``.

Already implemented are ``tukey_windowed``, ``hamming_windowed``,
``blackman_windowed``.

zero_pad
^^^^^^^^

``zero_pad(N)`` pads the ``Timeseries`` with zeros so that it has a total of N
points. If ``N`` is smaller than the number of points in the ``Timeseries``, or
if the ``Timeseries`` is not equispaced in time, the operation will fail.


The FrequencySeries methods
---------------------------

normalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Normalize the ``FrequencySeries`` so that it maximum amplitude is one.


low_pass, high_pass, band_pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``low_pass``, ``high_pass``, and ``band_pass`` apply standard filters to remove
some frequencies. In case the signal is complex, both positive and negative
frequencies are removed (e.g., ``high_pass(fmin)`` removes frequencies ``f``
so that ``abs(f) <= f``).

peaks, peaks_frequencies
^^^^^^^^^^^^^^^^^^^^^^^^

``peaks(amp_threshold)`` detects the peaks (local maxima) in the amplitude of
the spectrum that are larger than ``amp_threshold``. It returns a list of
tuples. The first element of the tuple is the frequency bin in which the maximum
is found, the second is a estimate obtained using a quadratic fit, and the third
is the actual value of the amplitude. ``peaks_frequencies(amp_threshold)`` is
like ``peaks(amp_threshold)`` but returns only the fitted frequencies.

Often, it is better to normalize the series, so that ``amp_threshold`` becomes a
percentual value of the the maximum peak.

Inverse Fourier transform (to_TimeSeries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using NumPy's ``fft``, return a ``TimeSeries`` that is the inverse Fourier
transform. IT is that ``to_TimeSeries()`` composed with ``to_FrequencySeries()``
is the identity with the exception of the domain of definition. The time domain
is from :math:`-1\slash (2 * \Delta f)` to :math:`1\slash (2 * \Delta f)`.

Occasionally signals that are supposed to be real are turned into complex with
imaginary part that is zero to machine precision.

inner_product, and overlap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`h1, h2` frequency series and :math:`S_n` spectral noise density,
the inner product is typically defined as
.. :math:

   `(h_1, h_2) = 4 \Re \int_{f_min}^{f_max} \frac{h_1 h_2^*}{S_n}`.

The method :py:meth:`~.inner_product` computes this quantity, possibly for a
network of detectors. If the noise is not provided, ``S_n`` will be fixed to
one. Alternatively, if the noise is a :py:class:`~.FrequencySeries`, the inner
product for that weighted with that noise will be computed. Alternatively, if
``noises`` is a list of :py:class:`~.FrequencySeries`, then we will assume that
the user wants to compute the network inner product:

.. :math:

   `(h_1, h_2)_{\textrm{network}} = \sum_{\mathrm{detectors}} (h_1, h_2)`

where each detector has its own noise curve. Internally, ``h_1``, ``h_2``, and
``S_n`` will be resampled to a common frequency interval with the number of
points of the series with fewest points. Hence, the accuracy of the computation
is determined by the accuracy of the series with fewest points.

The series are assumed to be zero outside the range of definition. So, if
``f_min`` or ``f_max`` are too large or too small, the effective parameter will
be determined by the series. By default, ``f_min=0`` and ``f_max=inf``.

With the inner product, one compute the overlap between two series:

.. :math:

   `\textrm{overlap} = (h_1, h_2) / \sqrt{(h_1, h_1)(h_2, h_2)}`

Again, this can be unweighted, or noise-weighted, or for a network of
detectors (if a list of noises is provided).

load_FrequencySeries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function can be used to load a file as a :py:mod:`~.FrequencySeries`. This
is particularly useful for noise curves. Internally, this function uses Numpy's
``loadtxt`` so, additional arguments can be passed directly to that method.

For noise curves, you can use :py:meth:`~.load_noise_curve` with the path of the
file. (This internally uses :py:meth:`~.load_FrequencySeries`).

Additional functions in :py:mod:`~.timeseries`
----------------------------------------------

:py:mod:`~.timeseries` has also some additional useful functions, described
here.

combine_ts
^^^^^^^^^^

``combine_ts`` takes a list of ``TimeSeries`` as input and combine them in a
single new ``TimeSeries`` with monotonically increasing time. ``combine_ts`` can
be called with ``prefer_late=True`` (default) or not. The difference between the
two is that when ``prefer_late=False`` data from the ``TimeSeries`` with smaller
``tmin`` (i.e., the previous checkpoint) is preferred, and the opposite is true
for ``prefer_late=True`` (i.e., the later checkpoint is used).

sample_common
^^^^^^^^^^^^^^^^^^

``resample_common`` takes a list of ``TimeSeries`` and resamples all of them to
the largest time interval covered by all timeseries, using regularly spaced
time. The number of sample points is the minimum over all time series.

time_at_maximum, time_at_minimum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often it is useful to know where is the peak of a signal (for example, for
gravitational waves). These methods return the time at which the absolute value of
the signal is maximum and minimum respectively.

remove_duplicate_iters
^^^^^^^^^^^^^^^^^^^^^^

This function takes two arrays ``t`` and ``y`` and remove overlapping segments
of time (such as, from checkpointing) returning a ``TimeSeries`` with
monotonically increasing times.

unfold_phase
^^^^^^^^^^^^^^^^^

In gravitational-wave astronomy the phase of a wave is typically unfolded so
that instead of going from :math:`0` to :math:`2\pi`, it is free to assume any
value so that the number of periodicities can be counted. ``unfold_phase`` takes
a signal and removes all the jumps of :math:`2\pi`. Optionally, provide a time
``t_of_zero_phase``, the value of the phase is offset so that it is zero when
the time is ``t_of_zero_phase``.
