TwoPunctures metadata
=============================================

The ``TwoPunctures`` thorn saves a metadata file called ``TwoPunctures.bbh``
which contains the parameters for the initial data. Sometimes, it is useful to
access these variables. If ``kuibit`` finds a ``TwoPunctures.bbh``, it will read
it and store the content in a :py:class:`~.TwoPuncturesDir` object. This is a
dictionary-like object that maps the metadata in the ``TwoPunctures.bbh`` to
their values.

For example,

.. code-block:: python

   # Assuming sd is a SimDir
   print(sd.twopunctures["initial-ADM-energy"])

This will print the ADM mass computed by ``TwoPunctures``. To find what fields
are available, you can use the ``keys`` method.
