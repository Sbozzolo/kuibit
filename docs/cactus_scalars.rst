Scalar data
==============================

Scalar output are common in several simulations, with the most notable example
being reductions (max, min, average, ...). The module :py:mod:`~.cactus_scalars`
(:ref:`series_ref:Reference on postcactus.cactus_scalars`) handles these
quantities.

What data can be read?_
-----------------------

:py:class:`~.CactusScalarASCII` reads files produced by ``CarpetASCII``. It
recognizes transparently ``tg`` and ``bz2`` compressed files and it works with
multiple variables in one file, or different files for each variable. In the
former case, :py:class:`~.CactusScalarASCII` reads the ``column format`` line in
the file and deduces the content. :py:class:`~.CactusScalarASCII` can return
a :py:class:`~.TimeSeries` with the time evolution of the various scalars.
