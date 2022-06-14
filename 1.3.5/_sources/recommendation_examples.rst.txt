Recommendations on how to use the examples
==================================================

``kuibit`` comes with lots of examples, most in the form of scripts. All the
examples are ready to be used for scientific applications. They are general
enough so that you can immediately start using them for your shell. For
instance, you can make quick plots without writing a single line of code. In
this page, we describe how to set-up the examples in such a way that takes
advantage of all the features implemented. You can follow along the steps,
starting from the folder where you with to install the examples.

0. Acquire the examples
---------------------------

Every time a new version of ``kuibit`` is released, a new archive with the most
updated examples is created by GitHub actions. This is available in the `GitHub
release page
<https://github.com/sbozzolo/kuibit/releases/latest/download/examples.tar.gz>`_.
This is the simplest way to download all the available examples.

You can grab the latest copy with

..  code-block:: sh

    curl -O https://github.com/sbozzolo/kuibit/releases/latest/download/examples.tar.gz

Unpack the archive:

..  code-block:: sh

    tar -xvzf examples.tar.gz


1. Set up PATHS
---------------

It is convenient to be able to access the codes from anywhere in the file
system. If you are not interested in this, you can simply call the scripts with
the full path, or copy them in the directories where you want to use them.

If you prefer being able to access to the codes from anywhere, you need to set
up some environment variables. For the scripts, this is the ``$PATH`` variable.
For ``bash`` and ``zsh``, you will have to add a line like this to your
``.bashrc`` or ``.zshrc``:

..  code-block:: sh

    export PATH="FOLDER/examples/bins:$PATH"

where ``FOLDER`` is the folder where you decided to install the examples.

Next, we need to set up the path for ``motionpicture`` (see, :doc:`quick
introduction to motionpicture <motionpicture>`). For that, we need to define the
variable ``MOPI_MOVIES_DIR``. In the same way, this can be achieved with the
following line:

..  code-block:: sh

    export MOPI_MOVIES_DIR="FOLDER/examples/mopi_movies"

2. Set up tab completion
------------------------------

The examples use `argcomplete <https://kislyuk.github.io/argcomplete/>`_ to
enable tab completions on the available flags. If you are not interested in tab
completion, you can ignore this section. Alternatively, for ``bash``, you can
run ``activate-global-python-argcomplete --user`` and start a new shell to enjoy
tab completion. For ``zsh``, you have enable support for ``bash`` completion
scripts.

..  code-block:: sh

    autoload -U bashcompinit
    bashcompinit

Then, you have register each single script:

..  code-block:: sh

    for f in $(ls "bins"); do eval "$(register-python-argcomplete $f)"; done

For ``fish``:

.. code-block:: sh

    for f in (ls "bins"); register-python-argcomplete --shell fish $f > ~/.config/fish/completions/$f.fish; end
