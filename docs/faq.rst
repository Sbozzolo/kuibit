Frequently Asked Questions
=============================================

Some of the attributes in SimDir are slow
-----------------------------------------

When you instantiate a :py:class:`~.SimDir`, ``kuibit`` will go through all the
files and organize them. Next, when you ask for specialize your request (e.g.,
you want to work with grid data), ``kuibit`` will isolate the relevant files and
analyze them more carefully. This can take a long time if you have several
thousands of files. In some cases, ``kuibit`` has to open each single file to
determine its content (e.g., ``kuibit`` scans the headers of your files to see
what variables are inside). Opening files has some system overhead that cannot be
reduced.

To solve this problem, you have to reduce the number of files in your directory.
There are two ways to do this:

1. you create a new folder filled with files obtained by concatenating the
   corresponding files across different simulations restarts. For example, if
   your simulation has 10 restarts, each one with the same file `rho.x.asc`, you
   can concatenate all these files in a new `rho.x.asc` in the new folder. This
   will decrease the file count by a factor of 10 (the number of restarts).
2. you create new folders only containing the data you want to study. Let us
   assume that you are only interested in gravitational wave data. You can
   create a new folder that follows the same structure of your simulation
   directory, and create links to the files you are interested in. Then, when
   you use ``kuibit`` from this second folder, there will be far fewer files to
   analyze.

merge_refinement_levels() is too slow and/or requires too much memory
---------------------------------------------------------------------

When working with grid data, it is tempting to deal with
:py:class:`~.HierarchicalGridData` using :py:meth:`~.merge_refinement_levels` so
that you do not have to deal with all the complexity of the various refinement
levels. However, what :py:meth:`~.merge_refinement_levels` does is to reduce
everything to the highest resolution. If you have several refinement levels in a
large grid, this would require hundreds of terabytes!

:py:meth:`~.merge_refinement_levels` is provided as a convenience function for
small simulations. For larger simulations, you have to work directly with
:py:class:`~.HierarchicalGridData` (which fully support all the various
mathematical operations and other various methods), until you want to plot the
result. When are ready to plot, you should use the method
:py:meth:`~.to_UniformGridData` instead of :py:meth:`~.merge_refinement_levels`.
With this function you can control the region where you want to focus and the
resolution that you want to work with. In this way, you can reduce the number of
computations needed and make the problem tractable.
