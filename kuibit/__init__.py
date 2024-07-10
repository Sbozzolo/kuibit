#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""kuibit provides utilities to post-process simulations performed with the
Einstein Toolkit (or any Cactus-based code).

"""

# skipcq: PY-W2000
from kuibit.simdir import SimDir  # noqa: 401

__version__ = "1.5.0"

__bibtex__ = """\
@article{kuibit,
       author = {{Bozzola}, Gabriele},
        title = "{kuibit: Analyzing Einstein Toolkit simulations with Python}",
      journal = {The Journal of Open Source Software},
     keywords = {numerical relativity, Python, Einstein Toolkit, astrophysics, Cactus, General Relativity and Quantum Cosmology, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2021,
        month = apr,
       volume = {6},
       number = {60},
          eid = {3099},
        pages = {3099},
          doi = {10.21105/joss.03099},
archivePrefix = {arXiv},
       eprint = {2104.06376},
 primaryClass = {gr-qc},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021JOSS....6.3099B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
