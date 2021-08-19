First steps with ``kuibit``
===========================

``kuibit`` is a powerful library to analyze simulations performed with the
`Einstein Toolkit <https://einsteintoolkit.org/>`_, and other ``Cactus``-based
codes. The package has several features and it can appear intimidating at first.
In this page, we introduce you to the most important concepts and recommend a
possible path for you to learn how to use the tool. Some of these topics are
covered in the YouTube series `Using kuibit
<https://www.youtube.com/playlist?list=PLIVVtc6RlFxpi3BiF6KTWd3z4TTQ5hY06>`_.

A basic knowledge of Python and the command-line is required to follow the
content of this page. However, you don't need to know much about the Einstein
Toolkit to be able to get started.

What is ``kuibit``?
-------------------

``kuibit`` is a Python library to support you in your science. You mostly
interact with ``kuibit`` by importing its modules into Python scripts, Jupyter
notebooks, REPLs, or anything that can interpret Python (at least version
3.6.1).

For example, you can check if you have ``kuibit`` correctly installed with the
following piece of code:

.. code-block:: python

   import kuibit
   print(kuibit.__version___)

You can save this snippet into a file and run it as ``python3
name_of_the_file.py``, or execute it as a cell in a Jupyter notebook, or
anything else that you might prefer.

``kuibit`` is designed to support all the possible workflows.

How to install ``kuibit``?
--------------------------

If you found that you did not have ``kuibit`` installed, you can easily fix the
problem by running in your shell

.. code-block:: sh

   pip3 install -U kuibit

This will ensure that you have latest stable version of ``kuibit`` available.
Now, if you try to run again the code above, it should print the version of
``kuibit`` installed.

Test out ``kuibit`` with the examples
-------------------------------------

``kuibit`` comes with a lot of :doc:`examples <index>` that are ready to be used
for science. Examples are provided as Python scripts that have to be executed
from the command-line. If you want to take full advantage of the examples, you
can install them system-wide following the :doc:`recommendations on how to use
the examples <recommendation_examples>` page. Alternatively, you can simply save
them in the directory you want to analyze.

You can do a lot of things with the examples. The complete list with a short
description is available `on GitHub
<https://github.com/Sbozzolo/kuibit/tree/master/examples>`_. Examples have also
their space in the documentation (see, :doc:`Scripts <index>` and :doc:`Movies
<index>`).

Here, we will assume you have a simulation in the folder ``my_sim`` (typically
it will contain the subdirectories ``output-0000``, ``output-0001``, and so on).
We will refer to this as the *simulation directory*, or the *data directory*
(shortened as *datadir*).

There are two categories of examples: scripts, and movies.

Scripts
^^^^^^^

Scripts are valid Python codes that use ``kuibit`` to achieve a goal: most of
the scripts are targeted towards generating a plot. All the examples share some
common features. In the following, we will consider ``plot_grid_var.py`` to
illustrate how examples work.

* The examples are command-line scripts. If your examples are in your ``PATH``, you
  can call them in your shell with

.. code-block:: sh

   plot_grid_var.py


  If you saved them in the folder,

.. code-block:: sh

   ./plot_grid_var.py

* The examples are designed to be as general and flexible as possible. For
  instance, they should work on any simulation data, and they provide a lot of
  options to configure them.

* The examples have the ``--help`` flag.

.. code-block:: sh

   plot_grid_var.py --help


This will print something like:

.. code-block:: text

   plot_grid_var.py plots a given grid function.

   By default, no interpolation is performed so the image may look pixelated.
   There are two available modes of interpolation. The first is activated
   with --multilinear-interpolation. With this, the data from the simulation
   is interpolated with a multilinear interpolation onto the plotting grid.
   This is accurate and uses all the information available, but it is slow.
   A second way to perform interpolation is passing a --interpolation-method
   argument (e.g., bicubic). With this, the plotting data is interpolated.
   This is much faster but it is not as accurate.

   [-h] [-c CONFIGFILE] [-v] [--datadir DATADIR] [--outdir OUTDIR]
   [--ignore-symlinks] [--pickle-file PICKLE_FILE] [--resolution RESOLUTION]
   [-x0 ORIGIN ORIGIN] [-x1 CORNER CORNER] [--plane {xy,xz,yz}] [--figname
   FIGNAME] [--fig-extension FIG_EXTENSION] [--tikz-clean-figure] [--ah-show]
   [--ah-color AH_COLOR] [--ah-edge-color AH_EDGE_COLOR] [--ah-alpha AH_ALPHA]
   [--ah-time-tolerance AH_TIME_TOLERANCE] --variable VARIABLE [--iteration
   ITERATION] [--multilinear-interpolate] [--interpolation-method
   INTERPOLATION_METHOD] [--colorbar] [--logscale] [--vmin VMIN] [--vmax VMAX]
   [--absolute]

   optional arguments:
   -h, --help            show this help message and exit
   -c CONFIGFILE, --configfile CONFIGFILE
   Config file path
   -v, --verbose         Enable verbose output
   --datadir DATADIR     Data directory
   --outdir OUTDIR       Output directory
   --ignore-symlinks     Ignore symlinks in the data directory
   --pickle-file PICKLE_FILE   Read/write SimDir to this file
   --resolution RESOLUTION     Resolution of the grid in number of points (default: 500)
   -x0 ORIGIN ORIGIN, --origin ORIGIN ORIGIN
   -x1 CORNER CORNER, --corner CORNER CORNER
   --plane {xy,xz,yz}    Plane to plot (default: xy)
   --figname FIGNAME     Name of the output figure (not including the extension).
   --fig-extension FIG_EXTENSION
   Extension of the output figure (default: png). [env var: KBIT_FIG_EXTENSION]
   --tikz-clean-figure   Reduce the size of the figure when saving to a TikZ file.
   --variable VARIABLE   Variable to plot.
   --iteration ITERATION
   Iteration to plot. If -1, the latest.
   --multilinear-interpolate
   Whether to interpolate to smooth data with multilinear interpolation before plotting.
   --interpolation-method INTERPOLATION_METHOD
   Interpolation method for the plot. See docs of np.imshow. (default: none)
   --colorbar            Whether to draw the color bar.
   --logscale            Whether to use log scale.
   --vmin VMIN           Minimum value of the variable. If logscale is True, this has to be the log.
   --vmax VMAX           Maximum value of the variable. If logscale is True, this has to be the log.
   --absolute            Whether to take the absolute value.

   Horizon options:
   --ah-show             Plot apparent horizons.
   --ah-color AH_COLOR   Color name for horizons (default is 'k').
   --ah-edge-color AH_EDGE_COLOR
   Color name for horizons boundary (default is 'w').
   --ah-alpha AH_ALPHA   Alpha (transparency) for apparent horizons (default: 1)
   --ah-time-tolerance AH_TIME_TOLERANCE
   Tolerance for matching horizon time [simulation units] (default is '0.1').

   Args that start with '--' (eg. -v) can also be set in a config file
   (specified via -c). Config file syntax allows: key=value, flag=true,
   stuff=[a,b,c] (for details, see syntax at https://goo.gl/R74nmi). If an arg
   is specified in more than one place, then commandline values override
   environment variables which override config file values which override
   defaults.


This is a lot to digest, so let's focus on the most important flags.

1. The header describes briefly what the script is supposed to do and
   discusses some peculiarities.

2. ``--datadir``: Where the data lives. In our case, it is the folder
   ``my_sim``. You can specify the top-level folder, or the specific subfolder
   (e.g., ``my_sim/output-0000``), if you know exactly where the iteration you
   are interested in lives. Specifying the subfolder speeds up the discovery
   algorithm (organizing the data across the different subdirectories). This can
   be very significant for large simulations (alternatively, you can use
   pickles, see later).

3. ``--outdir``: Where to save the output. By default, this is the location
   where the script is run. If ``--figname`` is not specified, the output will
   have a default name

4. ``--pickle-file``: Scanning a simulation directory and all its subdirectories
   is an expensive operation. If the same simulation is analyzed with different
   scripts, it is convenient to save some information to disk. This is saved as
   Python pickle file. Passing this flag means that the pickle file is used and
   the directories are not scanned. If there's a mismatch between what the
   pickle file contains and the actual directories, that would possibly lead to
   an error. Hence, it is best to use pickle file only on simulations that have
   completed, or the it is best to regenerate the pickle file every time the
   data changes. This can be done with the picklify utility.

5. ``--fig-extension``: By default, images are saves as pngs. If tikz is passed,
   the images are exported to LaTeX. This variable can be set with a
   environmental variable ``KBIT_FIG_EXTENSION``.

6. ``--iteration``: This is which iteration you want to plot among the ones
   available in your data. Other examples will help you find which iterations are
   available. If you don't specify this quantity, the latest iteration available
   will be used.

7. ``--variable``: This is the name of the grid function that you want to plot.
   This is the same name you would write in the par file or call in your thorn.
   Examples might be ``rho``, or ``alp``. In this case, 2D files are used (HDF5
   preferred), so, you need to have output those variables for the plane you are
   interested in.

8. Finally, the footer informs us that we can use configuration files. For
   instance, instead of passing variables via the command line, we can write a
   text file with the same information: e.g., instead of ``--vmin 1`` , we would
   create a text file with content ``vmin=1`` named, for instance ``plot.conf``,
   and pass ``--configfile plot.conf``. Configuration files and command-line
   arguments can be mixed, but command-line arguments will have the precedence.


Combining all the different flags, a possible invocation of the example would be:

.. code-block:: sh

   plot_grid_var.py --datadir my_sim --variable rho -x0 -100 -100 -x1 100 100
                    --resolution 500 --logscale --colorbar --outdir plots
                    --vmin -10 --vmax 0 --iteration 0 --plane xy
                    --interpolation-method bicubic --verbose

Movies
^^^^^^^^

The examples in ``kuibit`` use `motionpicture`_ to produce videos.
``motionpicture`` is a Python package that helps developers render movies from
single frames. See :doc:`A quick introduction to motionpicture <motionpicture>`
for more details. ``motionpicture`` requires movie files to work, and ``kuibit``
provides some. For example, to make a 2D movie of any grid function, you can use
the ``grid_var`` movie file. Then, the flags are similar to the ones discussed
in the previous section, with the difference that movies are produced with the
``mopi`` binary:

.. code-block:: sh

   mopi grid_var --datadir my_sim --variable rho -x0 -100 -100 -x1 100 100
                 --resolution 500 --logscale --colorbar --outdir plots
                  --vmin -10 --vmax 0 --iteration 0 --plane xy
                  --interpolation-method bicubic --verbose --parallel
                  --min-frame 0 --max-frame 10240 --fps 60

Here, we also added some options for ``mopi`` (which can be explored with ``mopi
--help``). In particular, ``--parallel`` ensures that all the cores on the
machine are used to render the various frames.

Reproduce the examples
----------------------

Now that you have run the examples, you can move to the next step, which is to
start using the library. A useful pedagogical avenue to learn about ``kuibit``
is to consider the examples as "solved problems". You can pick some of the
examples and try to reproduce the same result. For instance, you may want to try
to plot a 2D grid variable. To do that, you can read the relevant
:doc:`Tutorials <index>` and :doc:`Usage <index>` pages. Those will instruct you
about the details of ``kuibit`` as a library.

.. note::

   The examples aim to be general, so they contain some boilerplate and several
   if/else statements. These are not essential.

Script
^^^^^^

Let's walk through one example: let's try to reproduce ``plot_grid_var.py``.
Since we want to work with grid data, the relevant tutorials are the one on
:doc:`SimDir <tutorials/simdir>` and the one one :doc:`grid data
<tutorials/grid_data>`.

First, we need import the relevant modules. In this case, we are only going to
need :py:mod:`~.simdir` and :py:mod:`~.visualize_matplotlib`. We are also going
to import ``matplotlib``.

.. code-block::

   from kuibit import simdir sd
   from kuibit import visualize_matplotlib as viz
   import matplotlib.pyplot as plt

Next, we initialize a :py:class:`~.SimDir` object. This is how all the codes
start, since :py:class:`~.SimDir` is how we interface with the simulation.

.. code-block::

   s = SimDir("my_sim")

The page :doc:`Getting started with SimDir <simdir>` contain useful information
about this object.

Now, we can specify some quantities of interest, like: what variable/iteration
do we want to read, or what plane, and so on.

.. code-block::

   VAR = "rho"
   ITERATION = 1024
   PLANE = "xy"
   X0 = -100, -100
   X1 = 100, 100
   SHAPE = 500, 500
   LOGSCALE = True
   VMIN, VMAX = -10, 1

``X0`` and ``X1`` are respectively the lower and the topmost corner of the
region we want to plot, in computational units (the same units of the
simulation). ``LOGSCALE`` will specify if we want to use base-10 logarithm or
not, and ``VMIN``, ``VMAX`` define the range where we want to plot (in
log). ``SHAPE`` will be discussed in the next paragraph.

We can finally read the variable as :py:class:`~.HierarchicalGridData`. This is
a complex object containing all the various components and refinement levels.
This object cannot be plotted directly, but it needs to be resampled to a
:py:class:`~.UniformGridData`, which is a simpler object that contains a regular
grid and data defined on this grid. The variable ``SHAPE`` controls the
resolution of this grid.

.. code-block::

   reader = s.gridfunctions[PLANE][VAR]
   var = reader[ITERATION]

You can plot this quantity directly with :py:func:`~.plot_color`:

.. code-block::

   plot_color(var,
              shape=SHAPE,
              x0=X0,
              x1=X1,
              vmin=VMIN,
              vmax=VMAX,
              logscale=LOGSCALE)

You can use everything you know about ``matplotlib``. For example, you can add
a title to the plot:

.. code-block::

   plt.plot("This is my first plot")
   plt.savefig("plot.pdf")

This is (almost) the minimum code possible to plot any given iteration of any
given grid function. You should now try to run it and compare it with the output
with the example. Next, you can have a look at the code of example to see what
other options are available.

Movie
^^^^^^

Let us use ``motionpicture`` to make a movie out of this. See :doc:`A quick
introduction to motionpicture <motionpicture>` for more details.

To use ``mopi``, we first to write a *movie file*, which is just a regular
Python file that defines a class ``MOPIMovie`` with three methods. The first is
``__init__(self, args)_``, which takes a ``Namespace`` containing the
command-line arguments passed (we are not going to use any here). The
``__init___`` does all the preparatory work needed to generate frames. In this
case, we want to initialize the :py:class:`~.SimDir` and the ``reader``, which
are the common work needed to make a frame.

.. code-block::

   from kuibit import simdir sd
   from kuibit import visualize_matplotlib as viz
   import matplotlib.pyplot as plt

   class MOPIMovie():
       def __init__(self, args):

           VAR = "rho"
           PLANE = "xy"

           self.X0 = -100, -100
           self.X1 = 100, 100
           self.SHAPE = 500, 500
           self.LOGSCALE = True
           self.VMIN, self.VMAX = -10, 1

           self.reader = sim("my_sim").gridfunctions[PLANE][VAR]


We made ``X0``, ``X1``, ``SHAPE``, and ``reader`` attributes (with
``self.``) because we want to access them in the other methods. The second
method is ``get_frames(self)`` which defines the list of frames that compose
the movie. In this case, we are going to use the iterations available

.. code-block::

         def get_frames(self):
             return self.reader.available_iterations

Finally, we need a method ``make_frame(self, path, iteration)`` that plots one
frame. For that, we essentially copy what we have done with the previous script:

.. code-block::

         def make_frame(self, path, iteration):
             # We need to clear up pre-existing figures
             plt.clf()

             plot_color(self.reader[iteration],
                        shape=RESOLUTION,
                        x0=X0,
                        x1=X1,
                        vmin=self.VMIN,
                        vmax=self.VMAX,
                        logscale=self.LOGSCALE)

             # Alternatively
             # plot_color(self.reader,
             #            iteration=iteration,
             #            shape=RESOLUTION,
             #            x0=X0,
             #            x1=X1)

             plt.title(f"Iteration = {iteration}")

             plt.savefig(path)

This is it! Now you can save this as a file and run it with ``mopi -m file_name``.

You can make this script more flexible by adding command-line arguments, so that
you don't have to modify the file when you want to change parameters.

Glossary
--------

Here we collect vocabulary that you might find used in ``kuibit``.

* Datadir: where the output of simulation lives.
* Grid function: simulation data defined on the grid.
* :py:class:`~.HierarchicalGridData`: collection of components at possibly
  different refinements that form a grid with several levels.
* `motionpicture`_ (mopi): external Python program to render movies.
* Origin (corner): bottom left (top right) cell in a center-centered grid.
* :py:class:`~.SimDir`: fundamental interface to the data in the simulation.
* Outdir: where to save the output of an example.
* Pickle: binary file where a :py:class:`~.SimDir` can be saved.
* TikZ: package to render graphics in LaTeX. ``kuibit`` can optionally output in
  this format.
* :py:class:`~.UniformGridData`: data defined on a uniform grid.

.. _motionpicture: https://github.com/Sbozzolo/motionpicture

Still confused?
-----------------------

Feel free to ask questions in the Telegram group or send an email to
gabrielebozzola@email.arizona.edu.
