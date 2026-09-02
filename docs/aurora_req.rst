Requirements
============


Python requirements
-------------------

Aurora uses the latest Python-3 distribution and requires a modern Fortran compiler, available on most Unix systems. Additionally, the following packages are automatically installed (from PyPI) when installing Aurora:

  numpy scipy matplotlib pandas requests

Aurora has no dependency on `omfit_classes`. Reading EFIT gEQDSK files and fetching data from tokamak databases (MDS+) is left to the caller: Aurora accepts an already-processed equilibrium as a plain dictionary, so any magnetic-reconstruction or postprocessing package can be used.



Julia requirements
------------------

To run the Julia version of the code, Julia must be installed; see::

  https://julialang.org/downloads/

Everything else should be automatically handled by the Aurora installation (see :ref:`Installation`).
