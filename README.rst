Aurora: impurity transport, plasma-wall interaction, neutrals and radiation modeling
====================================================================================

.. warning::

   **This is a modified fork.**

   `Next-Step-Fusion/Aurora <https://github.com/Next-Step-Fusion/Aurora>`_ is a fork of
   `fsciortino/Aurora <https://github.com/fsciortino/Aurora>`_, trimmed for embedding the
   impurity-transport solver in an integrated modelling suite alongside a free-boundary
   equilibrium and transport solver. It is **not** a drop-in replacement for upstream Aurora.

   What differs from upstream:

   * **No OMFIT dependency.** ``omfit_classes`` and its transitive dependencies are gone.
     Aurora no longer reads EFIT g-files or fetches data from MDS+; it takes an
     already-processed equilibrium as a plain dictionary, or none at all.
   * **Python >= 3.11 and numpy >= 2.0 are required**, and the runtime dependency set is
     reduced to ``numpy, scipy, matplotlib, pandas, requests``.
   * **Removed subsystems:** the SOLPS-ITER (``solps``) and OEDGE/DIVIMP (``oedge``)
     post-processing wrappers, the KN1D neutral-transport wrapper (``kn1d``), edge Lyman-alpha
     neutral analysis (``neutrals``), the animation helper (``animate``), and the experimental
     Julia backend.
   * **Reduced examples.** Scripts requiring OMFIT, the Julia backend, the removed wrappers or
     device-specific web services were deleted. Those that remain need no equilibrium file.
   * **Added:** a self-contained element table (``aurora.elements``) replacing OMFIT's
     ``atomic_element``, and a baseline-locked regression suite (see ``tests/README.md``).

   **If you want the full Aurora toolbox, use upstream instead** --
   ``pip install aurorafusion`` installs upstream, not this fork. The PyPI and conda-forge
   packages, and the documentation at https://aurora-fusion.readthedocs.io, all describe
   upstream Aurora and do not reflect the changes listed above.

   All physics, the Fortran kernel and the original authorship are upstream's; see
   `Development`_ and ``USER_AGREEMENT.txt``.

Aurora is a package to simulate heavy-ion transport, plasma-wall interaction (PWI), neutrals and radiation in magnetically-confined plasmas. It includes a 1.5D impurity transport forward model for the plasma ions, thoroughly benchmarked with the widely-adopted STRAHL code, and a simple multi-reservoir particle balance model including neutrals recycling, pumping and interaction with the material surfaces of the simulated device. A simple interface to plot and process atomic and surface data for fusion plasmas makes it a convenient tool for spectroscopy, PWI and integrated modeling. It also offers routines to analyze neutral states of hydrogen isotopes from neutral beam injection. The spectroscopic and PWI calculations can be not only applied to the output of Aurora's own forward model, but also coupled with other 1D, 2D or 3D transport codes.

Aurora's code is written in Python 3 and Fortran 90.

Upstream documentation is available at https://aurora-fusion.readthedocs.io -- note it describes
upstream Aurora, not this fork. This fork ships no ``docs/`` directory; see the fork notes above,
``tests/README.md``, and the ``TASK*.md`` files at the repository root.


Development 
-----------

**Upstream Aurora** is developed and maintained by F. Sciortino (MPI-IPP) in collaboration with T. Odstrcil (GA), A. Zito (MPI-IPP), D. Fajardo (MPI-IPP), A. Cavallaro (MIT) and R. Reksoatmodjo (W&M), with support from O. Linder (MPI-IPP), C. Johnson (U. Auburn), D. Stanczak (IPPLM) and S. Smith (GA). The STRAHL documentation provided by R.Dux (MPI-IPP) was extremely helpful to guide the initial development of Aurora. All of the physics and the Fortran kernel in this fork are their work.

For upstream Aurora, get in touch at fsciortino-at-proximafusion.com or open a pull-request at
`fsciortino/Aurora <https://github.com/fsciortino/Aurora>`_. Anything of general use should go
upstream rather than staying in this fork.

**This fork** is maintained by Next Step Fusion for coupling the impurity-transport solver to a
free-boundary equilibrium and transport solver. Issues specific to the changes listed at the top
of this file belong at `Next-Step-Fusion/Aurora <https://github.com/Next-Step-Fusion/Aurora>`_.

Installation
------------

This fork is **not published to PyPI or conda-forge** -- ``pip install aurorafusion`` gives you
upstream Aurora. Install this fork from source:

    git clone https://github.com/Next-Step-Fusion/Aurora
    cd Aurora
    pip install .

Use ``pip install --editable .`` if you intend to modify the code.

Requirements:

* **Python >= 3.11**
* a Fortran compiler, CMake >= 3.17.2 and Ninja >= 1.10 (the 1.5D kernel is Fortran 90, built
  through f2py by scikit-build-core). These are **not** installed by pip -- provide them via your
  system package manager or container image.
* runtime dependencies ``numpy>=2.0.0, scipy>=1.13, matplotlib>=3.9, pandas>=2.2.2, requests``,
  installed automatically.

Verified on Python 3.11 (numpy 2.4, scipy 1.17) and Python 3.13 (numpy 2.5, scipy 1.18).

After installing, run the regression suite to check the build:

    pip install pytest
    python -m pytest tests/ -v

See ``tests/README.md`` for what it covers and the available options.


Atomic data
-----------

Aurora offers a simple interface to download, read, process and plot atomic data from the Atomic Data and Structure Analysis (ADAS) database, particularly through the OPEN-ADAS website: https://open.adas.ac.uk . ADAS data files can be fetched remotely and stored within the Aurora distribution directory, or users may choose to fetch ADAS files from a chosen, pre-existing directory by setting

    export AURORA_ADAS_DIR=my_adas_directory
    
within their Linux environment (or analogous). If an ADAS files that is not available in AURORA_ADAS_DIR is requested by a user, Aurora attempts to download it and store it there. If you are using a public installation of Aurora and you do not have write-access to the directory where Aurora is installed, make sure to set AURORA_ADAS_DIR to a directory where you do have write-access before starting.

Several ADAS formats can currently be managed -- please see the upstream documentation.

.. note::
   ``adas_files`` downloads into the installed package directory by default, which is a problem
   for containerised or read-only installations. Set ``AURORA_ADAS_DIR`` to a writable location,
   or vendor the handful of adf11 files you need.


Surface data
------------

Aurora also contains an interface to read and plot plasma-material interaction data, for the most fusion-relevant ion species and wall materials, namely concerning reflection, sputtering and implantation of plasma ions from/into wall materials. The data were generated with the TRIM.SP Monte Carlo program. 

Please contact the authors to request and/or suggest expansions of current capabilities.


License
-------

Aurora is distributed under the MIT License. The package is made open-source with the hope that this will speed up research on fusion energy and make further code development easier. However, we kindly ask that all users communicate to us their purposes, difficulties and successes with Aurora, so that we may support users as much as possible and grow the code further.

This fork inherits that licence and the accompanying ``USER_AGREEMENT.txt`` unchanged.


Citing Aurora
-------------

Please see the `User Agreement <https://github.com/fsciortino/Aurora/blob/master/USER_AGREEMENT.txt>`_.
Cite upstream Aurora, not this fork -- the physics being cited is theirs. 
