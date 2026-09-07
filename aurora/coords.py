# MIT License
#
# Copyright (c) 2021 Francesco Sciortino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np, sys, os
from scipy.interpolate import interp1d, RectBivariateSpline
import copy
from scipy.ndimage import map_coordinates
from scipy.interpolate import InterpolatedUnivariateSpline
from . import grids_utils


def get_rhop_RZ(R, Z, geqdsk):
    """Find rhop at every R,Z [m] based on the equilibrium in the geqdsk dictionary."""
    return RectBivariateSpline(
        geqdsk["AuxQuantities"]["Z"],
        geqdsk["AuxQuantities"]["R"],
        geqdsk["AuxQuantities"]["RHOpRZ"],
    ).ev(Z, R)


def vol_average(quant, rhop, geqdsk, method="fs"):
    """Calculate the volume average of the given radially-dependent quantity on a rhop grid.

    Parameters
    ----------
    quant : array, (space, ...)
        quantity that one wishes to volume-average. The first dimension must correspond to space,
        but other dimensions may be exist afterwards.
    rhop : array, (space,)
        Radial rhop coordinate in cm units.
    geqdsk : dict
        Processed EFIT geqdsk containing the magnetic geometry. Method 'fs' needs
        ``geqdsk['fluxSurfaces']['geo']['psin']`` and ``['vol']``; method
        'fluxsurfaces' additionally needs ``geqdsk['fluxSurfaces']`` to provide a
        ``volume_integral`` method.
    method : {'fs','fluxsurfaces'}
        Method to evaluate the volume average. 'fs' uses a simple cumulative sum in r_V
        coordinates and needs nothing beyond a plain dictionary; 'fluxsurfaces' delegates to the
        ``volume_integral`` method of the flux-surface object, if the supplied `geqdsk` carries one.
        The methods only slightly differ in their results. Note that 'fluxsurfaces' will fail if
        rhop extends beyond the LCFS, while method 'fs' can estimate volume averages also into the
        SOL. Default is method='fs'.

    Returns
    -------
    quant_vol_avg : array, (space, ...)
        Volume average of the quantity given as an input, in the same units as in the input.
        If extrapolation beyond the range available from EFIT volume averages over a shorter section
        of the radial grid will be attempted. This does not affect volume averages within the LCFS.
    """
    if np.max(rhop) > 1.0:
        print(
            "Input rhop goes beyond the LCFS! Results may not be meaningful (and can only be obtained via method=='fs')."
        )

    if geqdsk is None:
        raise ValueError(
            "vol_average requires a geqdsk. Aurora no longer fetches equilibria from MDS+; "
            "load or build the equilibrium yourself and pass it in as a dictionary."
        )

    if method == "fs":
        # obtain mapping between rhop and r_V coordinates
        rho_pol, r_V_ = grids_utils.get_rhopol_rvol_mapping(geqdsk)

        # find r_V corresponding to input rhop (NB: extrapolation in the far SOL should be used carefully)
        r_V = interp1d(rho_pol, r_V_, bounds_error=False)(rhop)

        # use convenient volume averaging in r_V coordinates
        if np.any(np.isnan(r_V)):
            print(
                "Ignoring all nan points! These may derive from an attempted extrapolation or from nan inputs"
            )

        vol_avg = rV_vol_average(quant[~np.isnan(r_V)], r_V[~np.isnan(r_V)])

    elif method in ("fluxsurfaces", "omfit"):
        # delegate to the volume_integral method of the flux-surface object, if there is one
        rhopp = np.sqrt(geqdsk["fluxSurfaces"]["geo"]["psin"])
        quantp = interp1d(rhop, quant, bounds_error=False, fill_value="extrapolate")(
            rhopp
        )
        vol_avg = geqdsk["fluxSurfaces"].volume_integral(quantp)

    else:
        raise ValueError("Input method for volume average could not be recognized")

    return vol_avg


def rV_vol_average(quant, r_V):
    r"""Calculate a volume average of the given radially-dependent quantity on a r_V grid.
    This function makes use of the fact that the r_V radial coordinate, defined as
    :math:`r_V = \sqrt{ V / (2 \pi^2 R_{axis} }`,
    maps shaped volumes onto a circular geometry, making volume averaging a trivial
    operation via
    :math:`\langle Q \rangle = \Sigma_i Q(r_i) 2 \pi \ \Delta r_V`
    where :math:`\Delta r_V` is the spacing between radial points in r_V.

    Note that if the input r_V coordinate is extended outside the LCFS,
    this function will return the effective volume average also in the SOL, since it is
    agnostic to the presence of the LCFS.

    Parameters
    ----------
    quant : array, (space, ...)
        quantity that one wishes to volume-average. The first dimension must correspond to r_V,
        but other dimensions may be exist afterwards.
    r_V : array, (space,)
        Radial r_V coordinate in cm units.

    Returns
    -------
    quant_vol_avg : array, (space, ...)
        Volume average of the quantity given as an input, in the same units as in the input
    """
    quant_vol_avg = (
        2.0 * np.cumsum(quant * r_V * np.diff(r_V, prepend=0.0)) / (r_V[-1] ** 2)
    )

    return quant_vol_avg


def rad_coord_transform(x, name_in, name_out, geqdsk):
    """Transform from one radial coordinate to another. Note that this coordinate conversion is only
    strictly valid inside of the LCFS. A number of common coordinate nomenclatures are accepted, but
    it is recommended to use one of the coordinate names indicated in the input descriptions below.

    Parameters
    ----------
    x: array or float
        input x coordinate
    name_in: str
        input x coordinate name ('rhon','rvol','rhop','rhov','Rmid','rmid','r/a')
    name_out: str
        input x coordinate ('rhon','psin','rvol', 'rhop','rhov','Rmid','rmid','r/a')
    geqdsk: dict
        Processed gEQDSK dictionary containing the magnetic geometry.

    Returns
    -------
    array
        Conversion of `x` input for the requested radial grid coordinate.
    """
    if name_in == name_out:
        return x
    x = copy.deepcopy(x)

    # avoid confusion with name conventions
    conventions = {
        "rvol": "rvol",
        "r_vol": "rvol",
        "r_V": "rvol",
        "rhon": "rhon",
        "psin": "psin",
        "Psin": "psin",
        "rho_tor": "rhon",
        "rho_pol": "rhop",
        "rhop": "rhop",
        "r/a": "r/a",
        "roa": "r/a",
        "rhov": "rhov",
        "rho_V": "rhov",
        "rho_v": "rhov",
        "Rmid": "Rmid",
        "R_mid": "Rmid",
        "rmid": "rmid",
        "r_mid": "rmid",
    }
    name_in = conventions[name_in]
    name_out = conventions[name_out]

    if "rvol" not in geqdsk["fluxSurfaces"]["geo"]:
        R0 = geqdsk["RMAXIS"]
        eq_vol = geqdsk["fluxSurfaces"]["geo"]["vol"]
        rvol = np.sqrt(eq_vol / (2 * np.pi**2 * R0))
        geqdsk["fluxSurfaces"]["geo"]["rvol"] = rvol

    # sqrt(norm. tor. flux)
    rhon_ref = geqdsk["fluxSurfaces"]["geo"]["rhon"]
    # norm. pol. flux
    psin_ref = geqdsk["fluxSurfaces"]["geo"]["psin"]
    # sqrt(norm. pol. flux)
    rhop_ref = np.sqrt(psin_ref)
    # volume radius
    rvol = geqdsk["fluxSurfaces"]["geo"]["rvol"]
    # R at midplane
    Rmid = geqdsk["fluxSurfaces"]["midplane"]["R"]
    # r at midplane
    R0 = geqdsk["fluxSurfaces"]["R0"]
    rmid = Rmid - R0

    # Interpolate to transform coordiantes
    if name_in == "rhon":
        coord_in = rhon_ref
    elif name_in == "psin":
        coord_in = psin_ref
    elif name_in == "rhop":
        coord_in = rhop_ref
    elif name_in == "rvol":
        coord_in = rvol
    elif name_in == "rhov":
        rvol_lcfs = np.interp(1, rhon_ref, rvol)
        coord_in = rvol / rvol_lcfs
    elif name_in == "Rmid":
        coord_in = rmid  # use rmid since it starts at 0.0, making interpolation easier
        x -= R0  # make x represent a rmid value
    elif name_in == "rmid":
        coord_in = rmid
    elif name_in == "r/a":
        rmid_lcfs = np.interp(1, rhon_ref, rmid)
        coord_in = rmid / rmid_lcfs
    else:
        raise ValueError("Input coordinate was not recognized!")

    if name_out == "rhon":
        coord_out = rhon_ref
    elif name_out == "psin":
        coord_out = psin_ref
    elif name_out == "rhop":
        coord_out = rhop_ref
    elif name_out == "rvol":
        coord_out = rvol
    elif name_out == "rhov":
        rvol_lcfs = np.interp(1, rhon_ref, rvol)
        coord_out = rvol / rvol_lcfs
    elif name_out == "Rmid":
        coord_out = rmid  # use rmid since it starts at 0.0, making interpolation easier
    elif name_out == "rmid":
        coord_out = rmid
    elif name_out == "r/a":
        rmid_lcfs = np.interp(1, rhon_ref, rmid)
        coord_out = rmid / rmid_lcfs
    else:
        raise ValueError("Output coordinate was not recognized!")

    # trick for better extrapolation
    ind0 = coord_in == 0
    out = np.interp(x, coord_in[~ind0], coord_out[~ind0] / coord_in[~ind0]) * x

    if (x == coord_in[0]).any() and np.sum(ind0):
        x0 = x == coord_in[0]
        out[x0] = coord_out[ind0]  # give exact magnetic axis

    if name_out == "Rmid":  # interpolation was done on rmid rather than Rmid
        out += R0

    return out




 
def rhoTheta2RZ(geqdsk, rho, theta, coord_in='rhop', n_line=201):
    '''Convert values of rho,theta into R,Z coordinates from a geqdsk dictionary.

    Parameters
    ----------
    geqdsk : dict
        Processed EFIT geqdsk containing the magnetic geometry. Must carry the keys
        'RMAXIS', 'ZMAXIS' and 'AuxQuantities' (with 'R', 'Z' and the rho map for
        `coord_in`). A file path is not accepted -- parse the g-file yourself.
    rho : np.ndarray
        Values of normalized radial coordinate to consider.
    theta : np.ndarray
        Values of poloidal angle coordinate to consider.
    coord_in : str
        Label describing the nature of the radial coordinate in use.
    n_line : int
        Number of points to discretize flux surface.

    Results
    -------
    R : np.array, (ntheta, nrho)
        Values of the major radius along flux surfaces.
    Z : np.array, (ntheta, nrho)
        Values of the vertical coordinate along flux surfaces.
    '''
    if isinstance(geqdsk, str):
        raise TypeError(
            "rhoTheta2RZ needs an already-processed geqdsk dictionary, not a file path. "
            "Aurora no longer reads g-files itself; parse the equilibrium with the tool of "
            "your choice and pass a dict carrying 'RMAXIS', 'ZMAXIS' and 'AuxQuantities'."
        )

    line_m = .9 # line length: 0.9 m
    t = np.linspace(0, 1, n_line)**.5*line_m
    c, s = np.cos(theta), np.sin(theta)

    tmpc = c[:,None]*t[None]
    tmps = s[:,None]*t[None]
    line_r = tmpc + geqdsk['RMAXIS']
    line_z = tmps + geqdsk['ZMAXIS']

    aux = geqdsk['AuxQuantities']
    Rmesh = aux['R']
    Zmesh = aux['Z']

    dr = (Rmesh[-1] - Rmesh[0])/(len(Rmesh) - 1)
    dz = (Zmesh[-1] - Zmesh[0])/(len(Zmesh) - 1)

    scaling = np.array([dr, dz])
    offset  = np.array([Rmesh[0], Zmesh[0]])

    coords = np.array((line_r, line_z))
    index = ((coords.T - offset) / scaling).T
    
    psin = map_coordinates(aux['PSIRZ_NORM'].T, index, mode='nearest',
                           order=2, prefilter=True)

    rho_line = rad_coord_transform(psin, 'psin', coord_in, geqdsk)

    rho = np.atleast_1d(rho)
    theta = np.atleast_1d(theta)
    R = np.empty((len(theta), len(rho)))
    Z = np.empty((len(theta), len(rho)))
 
    for k in range(len(theta)):
        
        monotonicity = np.cumprod(np.ediff1d(rho_line[k],1)>0)==1
        imax = np.argmax(rho_line[k, monotonicity])


        R[k] = InterpolatedUnivariateSpline(rho_line[k, :imax], line_r[k, :imax], k=2)(rho)
        Z[k] = InterpolatedUnivariateSpline(rho_line[k, :imax], line_z[k, :imax], k=2)(rho)
     
           
    return R,Z
