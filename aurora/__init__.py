name = "aurora"
__version__ = "3.0.6"

import numpy as np, os

from .core import *
from .atomic import *
from .adas_files import *
from .amdata import *
from .elements import *
from .surface import *
from .trim_files import *

from .source_utils import *
from .default_nml import *
from .grids_utils import *
from .coords import *
from .radiation import *

from .plot_tools import *

from .janev_smith_rates import *
from .nbi_neutrals import *

from .synth_diags import *

from .pwi import *

from .facit import *

aurora_dir = os.path.dirname(os.path.abspath(__file__))
