interactive_timertree.py
=============================================

``interactive_timertree.py`` reads the timing information output by ``Carpet``
(as XML files) are prepares a webpage with an interactive visualization on which
functions took most of the time.

The webpage has to be rendered in a webserver, which is automatically (but
optionally) started by ``interactive_timertree.py``. On a remote cluster, you
may need to copy the file locally.

.. literalinclude:: ../../../examples/bins/interactive_timertree.py
  :language: python
