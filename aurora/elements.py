"""Self-contained atomic-element data and helpers.

Aurora needs very little atomic bookkeeping: the nuclear charge :math:`Z` and the
mass number :math:`A` of the impurity and of the background ion. Historically this
came from ``omfit_classes.utils_math.atomic_element``, which made ``omfit_classes``
-- and its whole dependency tree -- a hard requirement of the solver. This module
replaces that lookup with a static table, so Aurora's core needs nothing beyond
numpy and scipy.

:py:data:`ELEMENTS` is a verbatim copy of the isotope data OMFIT shipped, in the
same order, and :py:func:`atomic_element` reproduces its query semantics, so
results are unchanged. Most of the data was originally gathered from
http://www.sisweb.com/referenc/source/exactmas.htm

For Aurora's own use, prefer the direct helper::

    from aurora.elements import get_element_Z_A
    Z, A = get_element_Z_A("W")      # -> (74, 184)

The OMFIT-compatible dictionary form is also available::

    from aurora.elements import atomic_element
    out = atomic_element(symbol="W")
    # {'W(184)': {'name': 'Tungsten', 'symbol': 'W', 'symbol_A': 'W(184)',
    #             'Z': 74, 'A': 184, 'mass': 183.950953,
    #             'abundance': 30.67, 'Z_ion': 74}}
"""

import copy
import re

__all__ = [
    "atomic_element",
    "get_element_Z_A",
    "get_element_symbol",
    "element_symbols",
    "ELEMENTS",
]

# (key, symbol_A, name, symbol, Z, A, mass [amu], natural abundance [%])
#
# The order is OMFIT's own, and is significant: a query matching several isotopes
# returns them in this order, so callers taking ``list(out.keys())[0]`` get the
# same record they used to. The electron is the one entry whose key ('elec(-1)')
# differs from its symbol_A ('e(0)'), and the D-T mixture is the one entry whose
# A is not an integer (2.5).
_TABLE = (
    ('elec(-1)', 'e(0)',     'Electron',       'e',   -1, 0,     0.000544241924816, 100),
    ('H(1)',     'H(1)',     'Hydrogen',       'H',    1, 1,     1.007825,          99.99),
    ('H(2)',     'H(2)',     'Hydrogen',       'H',    1, 2,     2.014102,          0.015),
    ('H(3)',     'H(3)',     'Hydrogen',       'H',    1, 3,     3.0160492,         0.0),
    ('D(2)',     'D(2)',     'Deuterium',      'D',    1, 2,     2.014101,          0.015),
    ('T(3)',     'T(3)',     'Tritium',        'T',    1, 3,     3.0160492,         0.0),
    ('DT(2.5)',  'DT(2.5)',  'Deuterium-Tritium', 'DT',   1, 2.5,   2.5150751,         0.0),
    ('He(3)',    'He(3)',    'Helium',         'He',   2, 3,     3.016029,          0.0001),
    ('He(4)',    'He(4)',    'Helium',         'He',   2, 4,     4.002603,          100.0),
    ('Li(6)',    'Li(6)',    'Lithium',        'Li',   3, 6,     6.015123,          7.42),
    ('Li(7)',    'Li(7)',    'Lithium',        'Li',   3, 7,     7.016005,          92.58),
    ('Be(9)',    'Be(9)',    'Beryllium',      'Be',   4, 9,     9.012183,          100.0),
    ('B(10)',    'B(10)',    'Boron',          'B',    5, 10,    10.012938,         19.8),
    ('B(11)',    'B(11)',    'Boron',          'B',    5, 11,    11.009305,         80.2),
    ('C(12)',    'C(12)',    'Carbon',         'C',    6, 12,    12.0,              98.9),
    ('C(13)',    'C(13)',    'Carbon',         'C',    6, 13,    13.003355,         1.1),
    ('N(14)',    'N(14)',    'Nitrogen',       'N',    7, 14,    14.003074,         99.63),
    ('N(15)',    'N(15)',    'Nitrogen',       'N',    7, 15,    15.000109,         0.37),
    ('O(16)',    'O(16)',    'Oxygen',         'O',    8, 16,    15.994915,         99.76),
    ('O(17)',    'O(17)',    'Oxygen',         'O',    8, 17,    16.999131,         0.038),
    ('O(18)',    'O(18)',    'Oxygen',         'O',    8, 18,    17.999159,         0.2),
    ('F(19)',    'F(19)',    'Fluorine',       'F',    9, 19,    18.998403,         100.0),
    ('Ne(20)',   'Ne(20)',   'Neon',           'Ne',  10, 20,    19.992439,         90.6),
    ('Ne(21)',   'Ne(21)',   'Neon',           'Ne',  10, 21,    20.993845,         0.26),
    ('Ne(22)',   'Ne(22)',   'Neon',           'Ne',  10, 22,    21.991384,         9.2),
    ('Na(23)',   'Na(23)',   'Sodium',         'Na',  11, 23,    22.98977,          100.0),
    ('Mg(24)',   'Mg(24)',   'Magnesium',      'Mg',  12, 24,    23.985045,         78.9),
    ('Mg(25)',   'Mg(25)',   'Magnesium',      'Mg',  12, 25,    24.985839,         10.0),
    ('Mg(26)',   'Mg(26)',   'Magnesium',      'Mg',  12, 26,    25.982595,         11.1),
    ('Al(27)',   'Al(27)',   'Aluminum',       'Al',  13, 27,    26.981541,         100.0),
    ('Si(28)',   'Si(28)',   'Silicon',        'Si',  14, 28,    27.976928,         92.23),
    ('Si(29)',   'Si(29)',   'Silicon',        'Si',  14, 29,    28.976496,         4.67),
    ('Si(30)',   'Si(30)',   'Silicon',        'Si',  14, 30,    29.973772,         3.1),
    ('P(31)',    'P(31)',    'Phosphorus',     'P',   15, 31,    30.973763,         100.0),
    ('S(32)',    'S(32)',    'Sulfur',         'S',   16, 32,    31.972072,         95.02),
    ('S(33)',    'S(33)',    'Sulfur',         'S',   16, 33,    32.971459,         0.75),
    ('S(34)',    'S(34)',    'Sulfur',         'S',   16, 34,    33.967868,         4.21),
    ('S(36)',    'S(36)',    'Sulfur',         'S',   16, 36,    35.967079,         0.02),
    ('Cl(35)',   'Cl(35)',   'Chlorine',       'Cl',  17, 35,    34.968853,         75.77),
    ('Cl(37)',   'Cl(37)',   'Chlorine',       'Cl',  17, 37,    36.965903,         24.23),
    ('Ar(36)',   'Ar(36)',   'Argon',          'Ar',  18, 36,    35.967546,         0.34),
    ('Ar(38)',   'Ar(38)',   'Argon',          'Ar',  18, 38,    37.962732,         0.063),
    ('Ar(40)',   'Ar(40)',   'Argon',          'Ar',  18, 40,    39.962383,         99.6),
    ('K(39)',    'K(39)',    'Potassium',      'K',   19, 39,    38.963708,         93.2),
    ('K(40)',    'K(40)',    'Potassium',      'K',   19, 40,    39.963999,         0.012),
    ('K(41)',    'K(41)',    'Potassium',      'K',   19, 41,    40.961825,         6.73),
    ('Ca(40)',   'Ca(40)',   'Calcium',        'Ca',  20, 40,    39.962591,         96.95),
    ('Ca(42)',   'Ca(42)',   'Calcium',        'Ca',  20, 42,    41.958622,         0.65),
    ('Ca(43)',   'Ca(43)',   'Calcium',        'Ca',  20, 43,    42.95877,          0.14),
    ('Ca(44)',   'Ca(44)',   'Calcium',        'Ca',  20, 44,    43.955485,         2.086),
    ('Ca(46)',   'Ca(46)',   'Calcium',        'Ca',  20, 46,    45.953689,         0.004),
    ('Ca(48)',   'Ca(48)',   'Calcium',        'Ca',  20, 48,    47.952532,         0.19),
    ('Sc(45)',   'Sc(45)',   'Scandium',       'Sc',  21, 45,    44.955914,         100.0),
    ('Ti(46)',   'Ti(46)',   'Titanium',       'Ti',  22, 46,    45.952633,         8.0),
    ('Ti(47)',   'Ti(47)',   'Titanium',       'Ti',  22, 47,    46.951765,         7.3),
    ('Ti(48)',   'Ti(48)',   'Titanium',       'Ti',  22, 48,    47.947947,         73.8),
    ('Ti(49)',   'Ti(49)',   'Titanium',       'Ti',  22, 49,    48.947871,         5.5),
    ('Ti(50)',   'Ti(50)',   'Titanium',       'Ti',  22, 50,    49.944786,         5.4),
    ('V(50)',    'V(50)',    'Vanadium',       'V',   23, 50,    49.947161,         0.25),
    ('V(51)',    'V(51)',    'Vanadium',       'V',   23, 51,    50.943963,         99.75),
    ('Cr(50)',   'Cr(50)',   'Chromium',       'Cr',  24, 50,    49.946046,         4.35),
    ('Cr(52)',   'Cr(52)',   'Chromium',       'Cr',  24, 52,    51.94051,          83.79),
    ('Cr(53)',   'Cr(53)',   'Chromium',       'Cr',  24, 53,    52.940651,         9.5),
    ('Cr(54)',   'Cr(54)',   'Chromium',       'Cr',  24, 54,    53.938882,         2.36),
    ('Mn(55)',   'Mn(55)',   'Manganese',      'Mn',  25, 55,    54.938046,         100.0),
    ('Fe(54)',   'Fe(54)',   'Iron',           'Fe',  26, 54,    53.939612,         5.8),
    ('Fe(56)',   'Fe(56)',   'Iron',           'Fe',  26, 56,    55.934939,         91.72),
    ('Fe(57)',   'Fe(57)',   'Iron',           'Fe',  26, 57,    56.935396,         2.2),
    ('Fe(58)',   'Fe(58)',   'Iron',           'Fe',  26, 58,    57.933278,         0.28),
    ('Co(59)',   'Co(59)',   'Cobalt',         'Co',  27, 59,    58.933198,         100.0),
    ('Ni(58)',   'Ni(58)',   'Nickel',         'Ni',  28, 58,    57.935347,         68.27),
    ('Ni(60)',   'Ni(60)',   'Nickel',         'Ni',  28, 60,    59.930789,         26.1),
    ('Ni(61)',   'Ni(61)',   'Nickel',         'Ni',  28, 61,    60.931059,         1.13),
    ('Ni(62)',   'Ni(62)',   'Nickel',         'Ni',  28, 62,    61.928346,         3.59),
    ('Ni(64)',   'Ni(64)',   'Nickel',         'Ni',  28, 64,    63.927968,         0.91),
    ('Cu(63)',   'Cu(63)',   'Copper',         'Cu',  29, 63,    62.929599,         69.17),
    ('Cu(65)',   'Cu(65)',   'Copper',         'Cu',  29, 65,    64.927792,         30.83),
    ('Zn(64)',   'Zn(64)',   'Zinc',           'Zn',  30, 64,    63.929145,         48.6),
    ('Zn(66)',   'Zn(66)',   'Zinc',           'Zn',  30, 66,    65.926035,         27.9),
    ('Zn(67)',   'Zn(67)',   'Zinc',           'Zn',  30, 67,    66.927129,         4.1),
    ('Zn(68)',   'Zn(68)',   'Zinc',           'Zn',  30, 68,    67.924846,         18.8),
    ('Zn(70)',   'Zn(70)',   'Zinc',           'Zn',  30, 70,    69.925325,         0.6),
    ('Ga(69)',   'Ga(69)',   'Gallium',        'Ga',  31, 69,    68.925581,         60.1),
    ('Ga(71)',   'Ga(71)',   'Gallium',        'Ga',  31, 71,    70.924701,         39.9),
    ('Ge(70)',   'Ge(70)',   'Germanium',      'Ge',  32, 70,    69.92425,          20.5),
    ('Ge(72)',   'Ge(72)',   'Germanium',      'Ge',  32, 72,    71.92208,          27.4),
    ('Ge(73)',   'Ge(73)',   'Germanium',      'Ge',  32, 73,    72.923464,         7.8),
    ('Ge(74)',   'Ge(74)',   'Germanium',      'Ge',  32, 74,    73.921179,         36.5),
    ('Ge(76)',   'Ge(76)',   'Germanium',      'Ge',  32, 76,    75.921403,         7.8),
    ('As(75)',   'As(75)',   'Arsenic',        'As',  33, 75,    74.921596,         100.0),
    ('Se(74)',   'Se(74)',   'Selenium',       'Se',  34, 74,    73.922477,         0.9),
    ('Se(76)',   'Se(76)',   'Selenium',       'Se',  34, 76,    75.919207,         9.0),
    ('Se(77)',   'Se(77)',   'Selenium',       'Se',  34, 77,    76.919908,         7.6),
    ('Se(78)',   'Se(78)',   'Selenium',       'Se',  34, 78,    77.917304,         23.5),
    ('Se(80)',   'Se(80)',   'Selenium',       'Se',  34, 80,    79.916521,         49.6),
    ('Se(82)',   'Se(82)',   'Selenium',       'Se',  34, 82,    81.916709,         9.4),
    ('Br(79)',   'Br(79)',   'Bromine',        'Br',  35, 79,    78.918336,         50.69),
    ('Br(81)',   'Br(81)',   'Bromine',        'Br',  35, 81,    80.91629,          49.31),
    ('Kr(78)',   'Kr(78)',   'Krypton',        'Kr',  36, 78,    77.920397,         0.35),
    ('Kr(80)',   'Kr(80)',   'Krypton',        'Kr',  36, 80,    79.916375,         2.25),
    ('Kr(82)',   'Kr(82)',   'Krypton',        'Kr',  36, 82,    81.913483,         11.6),
    ('Kr(83)',   'Kr(83)',   'Krypton',        'Kr',  36, 83,    82.914134,         11.5),
    ('Kr(84)',   'Kr(84)',   'Krypton',        'Kr',  36, 84,    83.911506,         57.0),
    ('Kr(86)',   'Kr(86)',   'Krypton',        'Kr',  36, 86,    85.910614,         17.3),
    ('Rb(85)',   'Rb(85)',   'Rubidium',       'Rb',  37, 85,    84.9118,           72.17),
    ('Rb(87)',   'Rb(87)',   'Rubidium',       'Rb',  37, 87,    86.909184,         27.84),
    ('Sr(84)',   'Sr(84)',   'Strontium',      'Sr',  38, 84,    83.913428,         0.56),
    ('Sr(86)',   'Sr(86)',   'Strontium',      'Sr',  38, 86,    85.909273,         9.86),
    ('Sr(87)',   'Sr(87)',   'Strontium',      'Sr',  38, 87,    86.908902,         7.0),
    ('Sr(88)',   'Sr(88)',   'Strontium',      'Sr',  38, 88,    87.905625,         82.58),
    ('Y(89)',    'Y(89)',    'Yttrium',        'Y',   39, 89,    88.905856,         100.0),
    ('Zr(90)',   'Zr(90)',   'Zirconium',      'Zr',  40, 90,    89.904708,         51.45),
    ('Zr(91)',   'Zr(91)',   'Zirconium',      'Zr',  40, 91,    90.905644,         11.27),
    ('Zr(92)',   'Zr(92)',   'Zirconium',      'Zr',  40, 92,    91.905039,         17.17),
    ('Zr(94)',   'Zr(94)',   'Zirconium',      'Zr',  40, 94,    93.906319,         17.33),
    ('Zr(96)',   'Zr(96)',   'Zirconium',      'Zr',  40, 96,    95.908272,         2.78),
    ('Nb(93)',   'Nb(93)',   'Niobium',        'Nb',  41, 93,    92.906378,         100.0),
    ('Mo(92)',   'Mo(92)',   'Molybdenum',     'Mo',  42, 92,    91.906809,         14.84),
    ('Mo(94)',   'Mo(94)',   'Molybdenum',     'Mo',  42, 94,    93.905086,         9.25),
    ('Mo(95)',   'Mo(95)',   'Molybdenum',     'Mo',  42, 95,    94.905838,         15.92),
    ('Mo(96)',   'Mo(96)',   'Molybdenum',     'Mo',  42, 96,    95.904676,         16.68),
    ('Mo(97)',   'Mo(97)',   'Molybdenum',     'Mo',  42, 97,    96.906018,         9.55),
    ('Mo(98)',   'Mo(98)',   'Molybdenum',     'Mo',  42, 98,    97.905405,         24.13),
    ('Mo(100)',  'Mo(100)',  'Molybdenum',     'Mo',  42, 100,   99.907473,         9.63),
    ('Ru(96)',   'Ru(96)',   'Ruthenium',      'Ru',  44, 96,    95.907596,         5.52),
    ('Ru(98)',   'Ru(98)',   'Ruthenium',      'Ru',  44, 98,    97.905287,         1.88),
    ('Ru(99)',   'Ru(99)',   'Ruthenium',      'Ru',  44, 99,    98.905937,         12.7),
    ('Ru(100)',  'Ru(100)',  'Ruthenium',      'Ru',  44, 100,   99.904218,         12.6),
    ('Ru(101)',  'Ru(101)',  'Ruthenium',      'Ru',  44, 101,   100.905581,        17.0),
    ('Ru(102)',  'Ru(102)',  'Ruthenium',      'Ru',  44, 102,   101.904348,        31.6),
    ('Ru(104)',  'Ru(104)',  'Ruthenium',      'Ru',  44, 104,   103.905422,        18.7),
    ('Rh(103)',  'Rh(103)',  'Rhodium',        'Rh',  45, 103,   102.905503,        100.0),
    ('Pd(102)',  'Pd(102)',  'Palladium',      'Pd',  46, 102,   101.905609,        1.02),
    ('Pd(104)',  'Pd(104)',  'Palladium',      'Pd',  46, 104,   103.904026,        11.14),
    ('Pd(105)',  'Pd(105)',  'Palladium',      'Pd',  46, 105,   104.905075,        22.33),
    ('Pd(106)',  'Pd(106)',  'Palladium',      'Pd',  46, 106,   105.903475,        27.33),
    ('Pd(108)',  'Pd(108)',  'Palladium',      'Pd',  46, 108,   107.903894,        26.46),
    ('Pd(110)',  'Pd(110)',  'Palladium',      'Pd',  46, 110,   109.905169,        11.72),
    ('Ag(107)',  'Ag(107)',  'Silver',         'Ag',  47, 107,   106.905095,        51.84),
    ('Ag(109)',  'Ag(109)',  'Silver',         'Ag',  47, 109,   108.904754,        48.16),
    ('Cd(106)',  'Cd(106)',  'Cadmium',        'Cd',  48, 106,   105.906461,        1.25),
    ('Cd(108)',  'Cd(108)',  'Cadmium',        'Cd',  48, 108,   107.904186,        0.89),
    ('Cd(110)',  'Cd(110)',  'Cadmium',        'Cd',  48, 110,   109.903007,        12.49),
    ('Cd(111)',  'Cd(111)',  'Cadmium',        'Cd',  48, 111,   110.904182,        12.8),
    ('Cd(112)',  'Cd(112)',  'Cadmium',        'Cd',  48, 112,   111.902761,        24.13),
    ('Cd(113)',  'Cd(113)',  'Cadmium',        'Cd',  48, 113,   112.904401,        12.22),
    ('Cd(114)',  'Cd(114)',  'Cadmium',        'Cd',  48, 114,   113.903361,        28.73),
    ('Cd(116)',  'Cd(116)',  'Cadmium',        'Cd',  48, 116,   115.904758,        7.49),
    ('In(113)',  'In(113)',  'Indium',         'In',  49, 113,   112.904056,        4.3),
    ('In(115)',  'In(115)',  'Indium',         'In',  49, 115,   114.903875,        95.7),
    ('Sn(112)',  'Sn(112)',  'Tin',            'Sn',  50, 112,   111.904826,        0.97),
    ('Sn(114)',  'Sn(114)',  'Tin',            'Sn',  50, 114,   113.902784,        0.65),
    ('Sn(115)',  'Sn(115)',  'Tin',            'Sn',  50, 115,   114.903348,        0.36),
    ('Sn(116)',  'Sn(116)',  'Tin',            'Sn',  50, 116,   115.901744,        14.7),
    ('Sn(117)',  'Sn(117)',  'Tin',            'Sn',  50, 117,   116.902954,        7.7),
    ('Sn(118)',  'Sn(118)',  'Tin',            'Sn',  50, 118,   117.901607,        24.3),
    ('Sn(119)',  'Sn(119)',  'Tin',            'Sn',  50, 119,   118.90331,         8.6),
    ('Sn(120)',  'Sn(120)',  'Tin',            'Sn',  50, 120,   119.902199,        32.4),
    ('Sn(122)',  'Sn(122)',  'Tin',            'Sn',  50, 122,   121.90344,         4.6),
    ('Sn(124)',  'Sn(124)',  'Tin',            'Sn',  50, 124,   123.905271,        5.6),
    ('Sb(121)',  'Sb(121)',  'Antimony',       'Sb',  51, 121,   120.903824,        57.3),
    ('Sb(123)',  'Sb(123)',  'Antimony',       'Sb',  51, 123,   122.904222,        42.7),
    ('Te(120)',  'Te(120)',  'Tellurium',      'Te',  52, 120,   119.904021,        0.096),
    ('Te(122)',  'Te(122)',  'Tellurium',      'Te',  52, 122,   121.903055,        2.6),
    ('Te(123)',  'Te(123)',  'Tellurium',      'Te',  52, 123,   122.904278,        0.91),
    ('Te(124)',  'Te(124)',  'Tellurium',      'Te',  52, 124,   123.902825,        4.82),
    ('Te(125)',  'Te(125)',  'Tellurium',      'Te',  52, 125,   124.904435,        7.14),
    ('Te(126)',  'Te(126)',  'Tellurium',      'Te',  52, 126,   125.90331,         18.95),
    ('Te(128)',  'Te(128)',  'Tellurium',      'Te',  52, 128,   127.904464,        31.69),
    ('Te(130)',  'Te(130)',  'Tellurium',      'Te',  52, 130,   129.906229,        33.8),
    ('I(127)',   'I(127)',   'Iodine',         'I',   53, 127,   126.904477,        100.0),
    ('Xe(124)',  'Xe(124)',  'Xenon',          'Xe',  54, 124,   123.905894,        0.1),
    ('Xe(126)',  'Xe(126)',  'Xenon',          'Xe',  54, 126,   125.904281,        0.09),
    ('Xe(128)',  'Xe(128)',  'Xenon',          'Xe',  54, 128,   127.903531,        1.91),
    ('Xe(129)',  'Xe(129)',  'Xenon',          'Xe',  54, 129,   128.90478,         26.4),
    ('Xe(130)',  'Xe(130)',  'Xenon',          'Xe',  54, 130,   129.90351,         4.1),
    ('Xe(131)',  'Xe(131)',  'Xenon',          'Xe',  54, 131,   130.905076,        21.2),
    ('Xe(132)',  'Xe(132)',  'Xenon',          'Xe',  54, 132,   131.904148,        26.9),
    ('Xe(134)',  'Xe(134)',  'Xenon',          'Xe',  54, 134,   133.905395,        10.4),
    ('Xe(136)',  'Xe(136)',  'Xenon',          'Xe',  54, 136,   135.907219,        8.9),
    ('Cs(133)',  'Cs(133)',  'Cesium',         'Cs',  55, 133,   132.905433,        100.0),
    ('Ba(130)',  'Ba(130)',  'Barium',         'Ba',  56, 130,   129.906277,        0.11),
    ('Ba(132)',  'Ba(132)',  'Barium',         'Ba',  56, 132,   131.905042,        0.1),
    ('Ba(134)',  'Ba(134)',  'Barium',         'Ba',  56, 134,   133.90449,         2.42),
    ('Ba(135)',  'Ba(135)',  'Barium',         'Ba',  56, 135,   134.905668,        6.59),
    ('Ba(136)',  'Ba(136)',  'Barium',         'Ba',  56, 136,   135.904556,        7.85),
    ('Ba(137)',  'Ba(137)',  'Barium',         'Ba',  56, 137,   136.905816,        11.23),
    ('Ba(138)',  'Ba(138)',  'Barium',         'Ba',  56, 138,   137.905236,        71.7),
    ('La(138)',  'La(138)',  'Lanthanum',      'La',  57, 138,   137.907114,        0.09),
    ('La(139)',  'La(139)',  'Lanthanum',      'La',  57, 139,   138.906355,        99.91),
    ('Ce(136)',  'Ce(136)',  'Cerium',         'Ce',  58, 136,   135.90714,         0.19),
    ('Ce(138)',  'Ce(138)',  'Cerium',         'Ce',  58, 138,   137.905996,        0.25),
    ('Ce(140)',  'Ce(140)',  'Cerium',         'Ce',  58, 140,   139.905442,        88.48),
    ('Ce(142)',  'Ce(142)',  'Cerium',         'Ce',  58, 142,   141.909249,        11.08),
    ('Pr(141)',  'Pr(141)',  'Praseodymium',   'Pr',  59, 141,   140.907657,        100.0),
    ('Nd(142)',  'Nd(142)',  'Neodymium',      'Nd',  60, 142,   141.907731,        27.13),
    ('Nd(143)',  'Nd(143)',  'Neodymium',      'Nd',  60, 143,   142.909823,        12.18),
    ('Nd(144)',  'Nd(144)',  'Neodymium',      'Nd',  60, 144,   143.910096,        23.8),
    ('Nd(145)',  'Nd(145)',  'Neodymium',      'Nd',  60, 145,   144.912582,        8.3),
    ('Nd(146)',  'Nd(146)',  'Neodymium',      'Nd',  60, 146,   145.913126,        17.19),
    ('Nd(148)',  'Nd(148)',  'Neodymium',      'Nd',  60, 148,   147.916901,        5.76),
    ('Nd(150)',  'Nd(150)',  'Neodymium',      'Nd',  60, 150,   149.9209,          5.64),
    ('Sm(144)',  'Sm(144)',  'Samarium',       'Sm',  62, 144,   143.912009,        3.1),
    ('Sm(147)',  'Sm(147)',  'Samarium',       'Sm',  62, 147,   146.914907,        15.0),
    ('Sm(148)',  'Sm(148)',  'Samarium',       'Sm',  62, 148,   147.914832,        11.3),
    ('Sm(149)',  'Sm(149)',  'Samarium',       'Sm',  62, 149,   148.917193,        13.8),
    ('Sm(150)',  'Sm(150)',  'Samarium',       'Sm',  62, 150,   149.917285,        7.4),
    ('Sm(152)',  'Sm(152)',  'Samarium',       'Sm',  62, 152,   151.919741,        26.7),
    ('Sm(154)',  'Sm(154)',  'Samarium',       'Sm',  62, 154,   153.922218,        22.7),
    ('Eu(151)',  'Eu(151)',  'Europium',       'Eu',  63, 151,   150.91986,         47.8),
    ('Eu(153)',  'Eu(153)',  'Europium',       'Eu',  63, 153,   152.921243,        52.2),
    ('Gd(152)',  'Gd(152)',  'Gadolinium',     'Gd',  64, 152,   151.919803,        0.2),
    ('Gd(154)',  'Gd(154)',  'Gadolinium',     'Gd',  64, 154,   153.920876,        2.18),
    ('Gd(155)',  'Gd(155)',  'Gadolinium',     'Gd',  64, 155,   154.822629,        14.8),
    ('Gd(156)',  'Gd(156)',  'Gadolinium',     'Gd',  64, 156,   155.92213,         20.47),
    ('Gd(157)',  'Gd(157)',  'Gadolinium',     'Gd',  64, 157,   156.923967,        15.65),
    ('Gd(158)',  'Gd(158)',  'Gadolinium',     'Gd',  64, 158,   157.924111,        24.84),
    ('Gd(160)',  'Gd(160)',  'Gadolinium',     'Gd',  64, 160,   159.927061,        21.86),
    ('Tb(159)',  'Tb(159)',  'Terbium',        'Tb',  65, 159,   158.92535,         100.0),
    ('Dy(156)',  'Dy(156)',  'Dysprosium',     'Dy',  66, 156,   155.924287,        0.06),
    ('Dy(158)',  'Dy(158)',  'Dysprosium',     'Dy',  66, 158,   157.924412,        0.1),
    ('Dy(160)',  'Dy(160)',  'Dysprosium',     'Dy',  66, 160,   159.925203,        2.34),
    ('Dy(161)',  'Dy(161)',  'Dysprosium',     'Dy',  66, 161,   160.926939,        18.9),
    ('Dy(162)',  'Dy(162)',  'Dysprosium',     'Dy',  66, 162,   161.926805,        25.5),
    ('Dy(163)',  'Dy(163)',  'Dysprosium',     'Dy',  66, 163,   162.928737,        24.9),
    ('Dy(164)',  'Dy(164)',  'Dysprosium',     'Dy',  66, 164,   163.929183,        28.2),
    ('Ho(165)',  'Ho(165)',  'Holmium',        'Ho',  67, 165,   164.930332,        100.0),
    ('Er(162)',  'Er(162)',  'Erbium',         'Er',  68, 162,   161.928787,        0.14),
    ('Er(164)',  'Er(164)',  'Erbium',         'Er',  68, 164,   163.929211,        1.61),
    ('Er(166)',  'Er(166)',  'Erbium',         'Er',  68, 166,   165.930305,        33.6),
    ('Er(167)',  'Er(167)',  'Erbium',         'Er',  68, 167,   166.932061,        22.95),
    ('Er(168)',  'Er(168)',  'Erbium',         'Er',  68, 168,   167.932383,        26.8),
    ('Er(170)',  'Er(170)',  'Erbium',         'Er',  68, 170,   169.935476,        14.9),
    ('Tm(169)',  'Tm(169)',  'Thulium',        'Tm',  69, 169,   168.934225,        100.0),
    ('Yb(168)',  'Yb(168)',  'Ytterbium',      'Yb',  70, 168,   167.933908,        0.13),
    ('Yb(170)',  'Yb(170)',  'Ytterbium',      'Yb',  70, 170,   169.934774,        3.05),
    ('Yb(171)',  'Yb(171)',  'Ytterbium',      'Yb',  70, 171,   170.936338,        14.3),
    ('Yb(172)',  'Yb(172)',  'Ytterbium',      'Yb',  70, 172,   171.936393,        21.9),
    ('Yb(173)',  'Yb(173)',  'Ytterbium',      'Yb',  70, 173,   172.938222,        16.12),
    ('Yb(174)',  'Yb(174)',  'Ytterbium',      'Yb',  70, 174,   173.938873,        31.8),
    ('Yb(176)',  'Yb(176)',  'Ytterbium',      'Yb',  70, 176,   175.942576,        12.7),
    ('Lu(175)',  'Lu(175)',  'Lutetium',       'Lu',  71, 175,   174.940785,        97.4),
    ('Lu(176)',  'Lu(176)',  'Lutetium',       'Lu',  71, 176,   175.942694,        2.6),
    ('Hf(174)',  'Hf(174)',  'Hafnium',        'Hf',  72, 174,   173.940065,        0.16),
    ('Hf(176)',  'Hf(176)',  'Hafnium',        'Hf',  72, 176,   175.94142,         5.2),
    ('Hf(177)',  'Hf(177)',  'Hafnium',        'Hf',  72, 177,   176.943233,        18.6),
    ('Hf(178)',  'Hf(178)',  'Hafnium',        'Hf',  72, 178,   177.94371,         27.1),
    ('Hf(179)',  'Hf(179)',  'Hafnium',        'Hf',  72, 179,   178.945827,        13.74),
    ('Hf(180)',  'Hf(180)',  'Hafnium',        'Hf',  72, 180,   179.946561,        35.2),
    ('Ta(180)',  'Ta(180)',  'Tantalum',       'Ta',  73, 180,   179.947489,        0.012),
    ('Ta(181)',  'Ta(181)',  'Tantalum',       'Ta',  73, 181,   180.948014,        99.99),
    ('W(180)',   'W(180)',   'Tungsten',       'W',   74, 180,   179.946727,        0.13),
    ('W(182)',   'W(182)',   'Tungsten',       'W',   74, 182,   181.948225,        26.3),
    ('W(183)',   'W(183)',   'Tungsten',       'W',   74, 183,   182.950245,        14.3),
    ('W(184)',   'W(184)',   'Tungsten',       'W',   74, 184,   183.950953,        30.67),
    ('W(186)',   'W(186)',   'Tungsten',       'W',   74, 186,   185.954377,        28.6),
    ('Re(185)',  'Re(185)',  'Rhenium',        'Re',  75, 185,   184.952977,        37.4),
    ('Re(187)',  'Re(187)',  'Rhenium',        'Re',  75, 187,   186.955765,        62.6),
    ('Os(184)',  'Os(184)',  'Osmium',         'Os',  76, 184,   183.952514,        0.02),
    ('Os(186)',  'Os(186)',  'Osmium',         'Os',  76, 186,   185.953852,        1.58),
    ('Os(187)',  'Os(187)',  'Osmium',         'Os',  76, 187,   186.955762,        1.6),
    ('Os(188)',  'Os(188)',  'Osmium',         'Os',  76, 188,   187.95585,         13.3),
    ('Os(189)',  'Os(189)',  'Osmium',         'Os',  76, 189,   188.958156,        16.1),
    ('Os(190)',  'Os(190)',  'Osmium',         'Os',  76, 190,   189.958455,        26.4),
    ('Os(192)',  'Os(192)',  'Osmium',         'Os',  76, 192,   191.961487,        41.0),
    ('Ir(191)',  'Ir(191)',  'Iridium',        'Ir',  77, 191,   190.960603,        37.3),
    ('Ir(193)',  'Ir(193)',  'Iridium',        'Ir',  77, 193,   192.962942,        62.7),
    ('Pt(190)',  'Pt(190)',  'Platinum',       'Pt',  78, 190,   189.959937,        0.01),
    ('Pt(192)',  'Pt(192)',  'Platinum',       'Pt',  78, 192,   191.961049,        0.79),
    ('Pt(194)',  'Pt(194)',  'Platinum',       'Pt',  78, 194,   193.962679,        32.9),
    ('Pt(195)',  'Pt(195)',  'Platinum',       'Pt',  78, 195,   194.964785,        33.8),
    ('Pt(196)',  'Pt(196)',  'Platinum',       'Pt',  78, 196,   195.964947,        25.3),
    ('Pt(198)',  'Pt(198)',  'Platinum',       'Pt',  78, 198,   197.967879,        7.2),
    ('Au(197)',  'Au(197)',  'Gold',           'Au',  79, 197,   196.96656,         100.0),
    ('Hg(196)',  'Hg(196)',  'Mercury',        'Hg',  80, 196,   195.965812,        0.15),
    ('Hg(198)',  'Hg(198)',  'Mercury',        'Hg',  80, 198,   197.96676,         10.1),
    ('Hg(199)',  'Hg(199)',  'Mercury',        'Hg',  80, 199,   198.968269,        17.0),
    ('Hg(200)',  'Hg(200)',  'Mercury',        'Hg',  80, 200,   199.968316,        23.1),
    ('Hg(201)',  'Hg(201)',  'Mercury',        'Hg',  80, 201,   200.970293,        13.2),
    ('Hg(202)',  'Hg(202)',  'Mercury',        'Hg',  80, 202,   201.970632,        29.65),
    ('Hg(204)',  'Hg(204)',  'Mercury',        'Hg',  80, 204,   203.973481,        6.8),
    ('Tl(203)',  'Tl(203)',  'Thallium',       'Tl',  81, 203,   202.972336,        29.52),
    ('Tl(205)',  'Tl(205)',  'Thallium',       'Tl',  81, 205,   204.97441,         70.48),
    ('Pb(204)',  'Pb(204)',  'Lead',           'Pb',  82, 204,   203.973037,        1.4),
    ('Pb(206)',  'Pb(206)',  'Lead',           'Pb',  82, 206,   205.974455,        24.1),
    ('Pb(207)',  'Pb(207)',  'Lead',           'Pb',  82, 207,   206.975885,        22.1),
    ('Pb(208)',  'Pb(208)',  'Lead',           'Pb',  82, 208,   207.976641,        52.4),
    ('Bi(209)',  'Bi(209)',  'Bismuth',        'Bi',  83, 209,   208.980388,        100.0),
    ('Th(232)',  'Th(232)',  'Thorium',        'Th',  90, 232,   232.038054,        100.0),
    ('U(234)',   'U(234)',   'Uranium',        'U',   92, 234,   234.040947,        0.006),
    ('U(235)',   'U(235)',   'Uranium',        'U',   92, 235,   235.043925,        0.72),
    ('U(238)',   'U(238)',   'Uranium',        'U',   92, 238,   238.050786,        99.27),
)

#: Isotope table, keyed as OMFIT keyed it (usually the ``symbol_A`` label, e.g.
#: ``ELEMENTS['C(12)']``). Each value is a dict with keys ``name``, ``symbol``,
#: ``symbol_A``, ``Z``, ``A``, ``mass`` and ``abundance``.
ELEMENTS = {
    _key: {
        "A": _A,
        "abundance": _ab,
        "name": _name,
        "symbol": _sym,
        "mass": _mass,
        "symbol_A": _sym_A,
        "Z": _Z,
    }
    for _key, _sym_A, _name, _sym, _Z, _A, _mass, _ab in _TABLE
}

# most abundant isotope of each atomic symbol -- the lookup Aurora actually needs
_MOST_ABUNDANT = {}
for _rec in ELEMENTS.values():
    _prev = _MOST_ABUNDANT.get(_rec["symbol"])
    if _prev is None or _prev["abundance"] < _rec["abundance"]:
        _MOST_ABUNDANT[_rec["symbol"]] = _rec
del _rec, _prev


def element_symbols():
    """Atomic symbols known to Aurora.

    Returns
    -------
    list of str
        Sorted atomic symbols, e.g. ``['Ag', 'Al', 'Ar', ...]``.
    """
    return sorted(_MOST_ABUNDANT)


def atomic_element(
    symbol_A=None,
    symbol=None,
    name=None,
    Z=None,
    Z_ion=None,
    mass=None,
    A=None,
    abundance=None,
    use_D_T=True,
    return_most_abundant=True,
):
    """Look up the atomic elements matching a query.

    Drop-in replacement for ``omfit_classes.utils_math.atomic_element``, backed by
    the static :py:data:`ELEMENTS` table instead of OMFIT.

    Parameters
    ----------
    symbol_A : str, optional
        Atomic symbol followed by the mass number in parentheses, e.g. ``'H(2)'``
        for deuterium.
    symbol : str, optional
        Atomic symbol. May be followed by the mass number (``'H2'`` for deuterium)
        or preceded by the mass number and followed by the ion charge number
        (``'2H1'``).
    name : str, optional
        Long name of the element, e.g. ``'Tungsten'``.
    Z : int, optional
        Atomic number, i.e. proton count in the nucleus.
    Z_ion : int, optional
        Charge number of the ion. If not given, ``Z_ion = Z`` is assumed.
    mass : float, optional
        Mass in amu. Matching on `A` is usually easier.
    A : int, optional
        Mass number, i.e. the mass rounded to the nearest integer.
    abundance : float, optional
        Natural abundance in percent.
    use_D_T : bool
        If True (default), report the heavy hydrogen isotopes as D and T rather
        than as H(2) and H(3).
    return_most_abundant : bool
        If True (default), collapse a query matching several isotopes of one
        element down to the most abundant isotope.

    Returns
    -------
    dict
        Matching records keyed by ``symbol_A``, each a dict with keys ``name``,
        ``symbol``, ``symbol_A``, ``Z``, ``A``, ``mass``, ``abundance`` and
        ``Z_ion``.

    Raises
    ------
    ValueError
        If `symbol` is malformed, or if no element satisfies the query.
    """
    if symbol:
        match = re.match(r"(\d*)([a-zA-Z]+)(\d*)$", symbol)
        if not match:
            raise ValueError(
                "Wrong form of symbol, expected H, H2, or 2H1 or similar, but got %s"
                % symbol
            )
        pre, symbol, post = match.groups()
        if pre == "" and post != "":
            A = int(post)
        elif pre != "" and post != "":
            A = int(pre)
            Z_ion = int(post)

    query = {
        "name": name,
        "symbol": symbol,
        "symbol_A": symbol_A,
        "Z": Z,
        "mass": mass,
        "A": A,
        "abundance": abundance,
    }
    # As in OMFIT, falsy values are dropped rather than matched, so e.g. Z=0 does
    # not filter at all.
    query = {k: v for k, v in query.items() if v}

    matches = {}
    for key, rec in ELEMENTS.items():
        if any(rec[field] != value for field, value in query.items()):
            continue
        matches[key] = copy.deepcopy(rec)
        matches[key]["Z_ion"] = int(rec["Z"]) if Z_ion is None else int(Z_ion)

    if return_most_abundant:
        best = {}
        for rec in matches.values():
            sym = rec["symbol"]
            if sym not in best or best[sym]["abundance"] < rec["abundance"]:
                best[sym] = rec
        matches = {rec["symbol_A"]: rec for rec in best.values()}

    # drop the redundant naming of the heavy hydrogen isotopes
    if len(matches) > 1:
        for key in list(matches):
            sym_A = matches[key]["symbol_A"]
            if not use_D_T and sym_A in ("D(2)", "T(3)"):
                del matches[key]
            elif use_D_T and sym_A in ("H(2)", "H(3)"):
                del matches[key]

    if not matches:
        raise ValueError(
            "No atomic element satisfies %s"
            % ", ".join("%s=%r" % kv for kv in query.items())
        )

    return matches


def get_element_Z_A(symbol):
    """Nuclear charge and mass number of an element, by atomic symbol.

    This is the lookup Aurora's solver needs. `A` is that of the most abundant
    natural isotope, e.g. 184 for tungsten -- matching what Aurora obtained from
    OMFIT before.

    Parameters
    ----------
    symbol : str
        Atomic symbol, e.g. ``'C'``, ``'W'``, ``'D'``. The other forms accepted by
        :py:func:`atomic_element`, such as ``'H2'``, work too.

    Returns
    -------
    Z : int
        Nuclear charge.
    A : int
        Mass number of the most abundant isotope.

    Raises
    ------
    ValueError
        If the symbol is not in the element table.
    """
    rec = _MOST_ABUNDANT.get(symbol)
    if rec is None:
        try:
            out = atomic_element(symbol=symbol)
        except ValueError:
            raise ValueError(
                "Unknown atomic symbol %r. Known symbols: %s"
                % (symbol, ", ".join(element_symbols()))
            )
        rec = out[list(out)[0]]
    return int(rec["Z"]), int(rec["A"])


def get_element_symbol(Z):
    """Atomic symbol of the element with nuclear charge `Z`.

    Parameters
    ----------
    Z : int
        Nuclear charge.

    Returns
    -------
    str
        Atomic symbol, e.g. ``'W'`` for ``Z=74``.

    Raises
    ------
    ValueError
        If no element with this nuclear charge is in the table.
    """
    for rec in _MOST_ABUNDANT.values():
        if rec["Z"] == int(Z):
            return rec["symbol"]
    raise ValueError("No element with nuclear charge Z=%r in the element table" % Z)
