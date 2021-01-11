Unit conversion and constants
=============================

``kuibit`` has a module, ``unitconv``, that with physical/astronomical
constants and with helper functions to convert between different unit systems.
:ref:`unitconv_ref:Reference on kuibit.unitconv`

Physical and astrophysical units
--------------------------------

``unitconv`` defined multiple constants in the SI unit system:

+---------------+---------------------------+
|   Variable    |         Constant          |
+===============+===========================+
|     C_SI      | Speed of light in vacuum  |
+---------------+---------------------------+
|     G_SI      |  Gravitational constant   |
+---------------+---------------------------+
|   M_SOL_SI    |        Solar mass         |
+---------------+---------------------------+
|   M_SUN_SI    |        Solar mass         |
+---------------+---------------------------+
|   PARSEC_SI   |          Parsec           |
+---------------+---------------------------+
| MEGAPARSEC_SI |        Megaparsec         |
+---------------+---------------------------+
| GIGAPARSEC_SI |        Gigaparsec         |
+---------------+---------------------------+
| LIGHTYEAR_SI  |        Light year         |
+---------------+---------------------------+
|     H0_SI     |     Hubble's constant     |
+---------------+---------------------------+

You can use these units as follows:

.. code-block:: python

    import kuibit.unitconv as uc

    print(f"1 Parsec is {uc.PARSEC_SI}")

Convert between geometrized units and SI
-----------------------------------------

Numerical relativity simulations are typically performed in geometrized units
with :math:`G = c = M = 1`, where :math:`M` is some mass scale. Often, we need
to convert these units to physical units. ``unitconv`` provides to tools for
that. The class ``Units`` is defined in ``unitconv``. Objects of the type
``Units`` are initialized providing a length, time and mass scales, then derived
units are automatically computed. For geometrized units, the simplest way to
perform unit conversion is initializing an ``Units`` object with
``geom_umass_msun``:

.. code-block:: python

    import kuibit.unitconv as uc

    # CU stands for Computational Units
    # Here we initialize a Units object for geometrized units with M = 65 M_sun
    CU = uc.geom_umass_msun(65)

The object ``CU`` can now convert from geometrized units to SI, for instance

.. code-block:: python

    d = 100 # M
    d_SI = d * CU.length

    energy = 5 # M
    energy_SI = energy * CU.energy

In case you need to use different unit systems you can instantiate directly a
``Units`` providing the length, time, and mass scales.

.. code-block:: python

    CGS = uc.Units(1e-2, 1, 1)

The functions ``geom_umass(SCALE)`` and ``geom_ulength(SCALE)`` return ``Units``
objects in which mass (or length) are set to ``SCALE``. The difference between
``geom_umass`` and ``geom_umass_msun`` is that the latter assumes that ``SCALE``
is in solar mass, the former in kilograms.

``Units`` objects know of a lot of quantities:

+----------------+---------------------------+
|    Variable    |        Dimensions         |
+================+===========================+
|     length     |            [L]            |
+----------------+---------------------------+
|      time      |            [T]            |
+----------------+---------------------------+
|      mass      |            [M]            |
+----------------+---------------------------+
|      freq      |           1/[T]           |
+----------------+---------------------------+
|    velocity    |          [L]/[T]          |
+----------------+---------------------------+
|     accel      |         [L]/[T]^2         |
+----------------+---------------------------+
|     force      |       [M][L]/[T]^2        |
+----------------+---------------------------+
|      area      |           [L]^2           |
+----------------+---------------------------+
|     volume     |           [L]^3           |
+----------------+---------------------------+
|    density     |         [M]/[L]^3         |
+----------------+---------------------------+
|    pressure    |      [M]/([L][T]^2)       |
+----------------+---------------------------+
|     power      |      [M][L]^2/[T]^3       |
+----------------+---------------------------+
|     energy     |      [M][L]^2/[T]^2       |
+----------------+---------------------------+
| energy_density |      [M]/([L][T]^2)       |
+----------------+---------------------------+
| angular_moment |       [M][L]^2/[T]        |
+----------------+---------------------------+
| moment_inertia |         [M][L]^2          |
+----------------+---------------------------+
