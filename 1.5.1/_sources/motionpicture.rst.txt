A quick introduction to motionpicture
=====================================

`motionpicture`_ is a Python library that takes care of all the infrastructure
needed to render frames and animate them into a video. When you install
``kuibit``, ``motionpicture`` will be automatically installed. With this page,
you will learn how the basics of how use ``motionpicture``.

.. _motionpicture: https://github.com/Sbozzolo/motionpicture

Overall considerations
----------------------

The key ingredient in ``motionpicture`` are *movie files*. These are Python
files that specify how to render one frame and what constitute a frame (e.g.,
one iteration = one frame). ``kuibit`` comes with a series of movie files for
common videos (e.g., 2D plots of a given grid function) which are ready to be
used. Later in this document, we will discuss how to write movie files.

The second important element in ``motionpicture`` is ``mopi``. ``mopi`` is a
command-line executable that takes as argument a movie file and renders the
frames and the videos. ``mopi`` has a lot of options that can be explored with
the ``--help`` flag.

``motionpicture`` requires ``ffmpeg``, which has to be installed separately.

Animating frames
----------------------

At level zero, ``motionpicture`` is a tool to glue together frames. Suppose you
have a folder ``frames`` full of files ``0000.png``, ``0001.png``, ..., and you
want to make a video out of them. You can use ``mopi`` for that.

.. code-block:: bash

   $ mopi --only-render-movie --frame-name-format '%04d.png' --outdir frames --movie-name my_video

This will create a file ``my_video.mp4`` with the specified frames. The advantage
of using ``mopi`` for this task is that it comes with a lot of options that can be
easily explored in the ``--help``. For example,

.. code-block:: bash

   $ mopi --only-render-movie --frame-name-format '%04d.png' --outdir frames --movie-name my_video \
          --fps 60 --extension webm --title "My video" --comment "This is my first video with mopi" \
          --author "Me"

Making and animating frames
---------------------------

Suppose you want to make frames, and then animate them. This is the core
functionality of ``motionpicture``, and the main benefits of using the tool is
that you have to worry only about making one single frame, and everything else
is taken care of (including parallelization)


``mopi`` can take as input a Python file which defines a class ``MOPIMovie``
which contains methods to identify what is a frame and how to draw it. Such
files can be specified with the flag ``-m``, or they can be read from a folder
defined by the environment variable ``MOPI_MOVIES_DIR``. ``mopi`` will load the
chosen movie and will expose all the command-line options that are defined
there. For instance, if you defined the environment variable ``MOPI_MOVIES_DIR``
and ``grid_var`` is in that folder, then ``mopi grid_var --help`` will look like
this:

.. code-block:: bash

    usage: mopi [-m MOVIE_FILE] [-c CONFIG] [--movies-dir MOVIES_DIR] [-o OUTDIR] [--snapshot SNAPSHOT] [--disable-progress-bar] [--parallel]
                [--num-workers NUM_WORKERS] [--only-render-movie] [--frame-name-format FRAME_NAME_FORMAT] [-v] [-h] [--min-frame MIN_FRAME]
                [--max-frame MAX_FRAME] [--frames-every FRAMES_EVERY] [--movie-name MOVIE_NAME] [--extension EXTENSION] [--fps FPS]
                [--author AUTHOR] [--title TITLE] [--comment COMMENT] [--datadir DATADIR] [--resolution RESOLUTION] [-x0 ORIGIN ORIGIN]
                [-x1 CORNER CORNER] [--plane {xy,xz,yz}] [--figname FIGNAME] [--fig-extension FIG_EXTENSION] [--ah-show] [--ah-color AH_COLOR]
                [--ah-edge-color AH_EDGE_COLOR] [--ah-alpha AH_ALPHA] [--ah-time-tolerance AH_TIME_TOLERANCE] --variable VARIABLE
                [--multilinear-interpolate] [--interpolation-method INTERPOLATION_METHOD] [--colorbar] [--logscale] [--vmin VMIN] [--vmax VMAX]
                [--absolute]
                [movie]

    Make a video specifying all the details using command-line arguments. To use this utility, you have to specify a movie. This code will look for movies in the MOPI_MOVIES_DIR, which you can customize. To select one of these movies, just pass the file name as first argument. Alternatively, you can pass the argument -m and specify a file.

    General options:
      movie                 Movie to render among the ones found in MOPI_MOVIES_DIR. See bottom of the help message for list.
      -m MOVIE_FILE, --movie-file MOVIE_FILE
                            Path of the movie file.
      -c CONFIG, --config CONFIG
                            Config file path
      --movies-dir MOVIES_DIR
                            Folder where to look form movies.   [env var: MOPI_MOVIES_DIR]
      -o OUTDIR, --outdir OUTDIR
                            Output directory for frames and video.
      --snapshot SNAPSHOT   Only produce the specified snapshot (useful for testing).
      --disable-progress-bar
                            Do not display the progress bar when generating frames.
      --parallel            Render frames in parallel.
      --num-workers NUM_WORKERS
                            Number of cores to use (default: 8).
      --only-render-movie   Do not generate frames but only render the final video.
      --frame-name-format FRAME_NAME_FORMAT
                            If only-render-movie is set, use this C-style frame name format instead of computing it. For example, '%04d.png' will assemble a video with frames with names 0000.png, 0001.png, and so on, as found in the outdir folder.
      -v, --verbose         Enable verbose output.
      -h, --help            Show this help message and exit.

    Frame selection:
      --min-frame MIN_FRAME
                            Do not render frames before this one.
      --max-frame MAX_FRAME
                            Do not render frames after this one.
      --frames-every FRAMES_EVERY
                            Render a frame every N (default: render all the possible frames).

    Video rendering options:
      --movie-name MOVIE_NAME
                            Name of output video file, without extension (default: video).
      --extension EXTENSION
                            File extension of the video (default: mp4).
      --fps FPS             Frames-per-second of the video (default: 25).
      --author AUTHOR       Author metadata in the final video.
      --title TITLE         Title metadata in the final video.
      --comment COMMENT     Comment metadata in the final video.

    Movie custom options:
      --datadir DATADIR     Data directory.
      --resolution RESOLUTION
                            Resolution of the grid in number of points (default: 500)
      -x0 ORIGIN ORIGIN, --origin ORIGIN ORIGIN
      -x1 CORNER CORNER, --corner CORNER CORNER
      --plane {xy,xz,yz}    Plane to plot (default: xy)
      --figname FIGNAME     Name of the output figure (not including the extension).
      --fig-extension FIG_EXTENSION
                            Extension of the output figure (default: png).   [env var: KBIT_FIG_EXTENSION]
      --variable VARIABLE   Variable to plot.
      --multilinear-interpolate
                            Whether to interpolate to smooth data with multilinear interpolation before plotting.
      --interpolation-method INTERPOLATION_METHOD
                            Interpolation method for the plot. See docs of np.imshow. (default: none)
      --colorbar            Whether to draw the color bar.
      --logscale            Whether to use log scale.
      --vmin VMIN           Minimum value of the variable. If logscale is True, this has to be the log.
      --vmax VMAX           Maximum value of the variable. If logscale is True, this has to be the log.
      --absolute            Whether to take the absolute value.

    No movies found in the MOPI_MOVIES_DIR (.)

    Args that start with '--' (eg. -m) can also be set in a config file (specified via -c). Config file syntax allows: key=value, flag=true,
    stuff=[a,b,c] (for details, see syntax at https://goo.gl/R74nmi). If an arg is specified in more than one place, then commandline values
    override environment variables which override config file values which override defaults.


With this, you can now make a 2D movie of any given grid function of any given
simulations with parameters you can control via command-line. One of the most
useful option is ``--parallel`` to render frames using all the cores of the
machine.

Let us have a look at a simplified version of ``grid_var``:

.. code-block:: python

    from kuibit import argparse_helper as kah
    from kuibit.simdir import SimDir
    from kuibit.visualize_matplotlib import (
        plot_color,
        save,
    )


    def mopi_add_custom_options(parser):
        # These are the custom options that mopi will see.

        parser.add_argument("--datadir", default=".", help="Data directory.")
        kah.add_grid_to_parser(parser, dimensions=2)
        kah.add_figure_to_parser(parser)

        parser.add_argument(
            "--variable", type=str, required=True, help="Variable to plot."
        )

    class MOPIMovie:
        def __init__(self, args):
            # Here we initialize all the objects that we need for all the frames.
            # All the expensive stuff has to be done here.

            self.sim = SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks)
            self.x0, self.x1, self.res = args.origin, args.corner, args.resolution
            self.shape = [self.res, self.res]
            self.reader = self.sim.gridfunctions[args.plane]
            self.var = self.reader[args.variable]

            self.iterations = self.var.available_iterations

        def get_frames(self):
            # Here we define what is a "frame" (it is one iteration). This function
            # has to return an iterable on what we want to make frames of.
            return self.iterations

        def make_frame(self, path, iteration):
            # Here we plot a frame. This function has to take the output path and
            # the identifier of what is a frame (in this case, an iteration).

            plot_color(
                self.var[iteration],
                x0=self.x0,
                x1=self.x1,
                shape=self.shape,
                xlabel=self.args.plane[0],
                ylabel=self.args.plane[1],
                resample=self.args.multilinear_interpolate,
                colorbar=self.args.colorbar,
                logscale=self.args.logscale,
                vmin=self.args.vmin,
                vmax=self.args.vmax,
                label=label,
                interpolation=self.args.interpolation_method,
            )

            save(path)

As you see, movie files are relatively simple. They have one (optional) function
``mopi_add_custom_options`` that defines the command-line options, and a class
``MOPIMovie``. This class has three methods: a ``__init__``, which is run once
and sets up the data that is used by all the frames, a ``get_frames``, which
defines what is a frame (here we used iteration, but it could be time, or
anything else), and a method ``make_frame`` which produces a given frame. Read
the `official documentation`_ for a detailed description on how to write movie
files.

.. _official documentation: https://github.com/Sbozzolo/motionpicture
