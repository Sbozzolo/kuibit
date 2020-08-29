Working with multipolar decompositions
======================================

Gravitational waves are typically studied in terms of their multipolar
decompositions :math:`(l, m)`. In Einstein Toolkit, ``Multipoles`` is
responsable for computing these quantities, which can be read and analyzed by
:py:mod:`~.cactus_multipoles`. Since the main application is gravitational
waves, we will use the word "detector" to mean "radius of the sphere where the
multipoles are computed".
