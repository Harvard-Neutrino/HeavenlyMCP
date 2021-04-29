from sklearn.utils import check_random_state
import numpy as np
from constants import *
from numpy import linalg as LA

def WriteToF2KFormat(MCPEvents, output_filename, n_write = None):
    """
        event_dictionary: dictionary of events
        output_filename: file name where the data will be written to
        n_write: number of events to serialize.
        This format output can be passed to PPC to propagate the photons.
        Details of the format can be found here:
        https://www.zeuthen.desy.de/~steffenp/f2000/
        In a nutshell this format is as follows:
        
        TR int int name x y z theta phi length energy time
        
        - TR stands for track and is a character constant.
        - The first two integer values are not used by ppc and are just for book keeping.
        - The name column specifies the track type.
            Possible values are: “amu+”, “amu-” and “amu” for muons, “delta”, “brems”, “epair”, “e+”, “e-” and “e” for electromagnetic cascades and “munu” and “hadr” for hadronic cascades. x, y and z are the vector components of the track’s initial position in meters.
        - theta and phi is the track’s theta and phi angle in degree, respectively.
            length is the length of the track in meter. 
            It is only required for muons because cascades are treated as point-like sources.
            energy is the track’s initial energy in GeV. 
        - time is the track’s initial time in nanoseconds.
    """
    f = open(output_filename, 'w')
    if n_write is None:
             n_write = len(MCPEvents)
    for i, (costh, phi, E_mCP, iposition, losses) in enumerate(MCPEvents, n_write):
        total_loss = np.sum([len(losses[loss]) for loss in losses])
        if(total_loss == 0):
            continue
        f.write("EM {i} 1 {depth_in_meters} 0 0 0 \n".format(i = i, depth_in_meters=IceCubeCenterDepth*km_to_m))
        f.write("MC E_mCP {E_mCP} x {x} y {y} z {z} theta {theta} phi {phi}\n".format(
            E_mCP = E_mCP,
            x = iposition[0],
            y = iposition[1],
            z = iposition[2],
            theta = np.arccos(costh),
            phi = phi))
        for ii, type_of_loss in enumerate(losses.keys()):
            xxx = []
            for loss in losses[type_of_loss]:
                wretch = GetCartesianPoint(np.arccos(costh),phi,LA.norm(loss[1]))
                wretch = np.insert(wretch,len(wretch),loss[0], axis=0)
                x = wretch
                magic_format = "TR {i} 0 {type_of_loss} {x} {y} {z} {theta} {phi} 0 {ee} {t} \n".format(
                    i = i,
                    type_of_loss = TypeLossNameConverter(type_of_loss),
                    x=x[0]*cm_to_m,
                    y=x[1]*cm_to_m,
                    z=x[2]*cm_to_m,
                    theta=np.arccos(costh),
                    phi=phi,
                    ee=10**x[3]*MeV_to_GeV,
                    t=0)
                f.write(magic_format)
        f.write("EE\n")
    f.close()

def GetCartesianPoint(theta,phi,r):
    """
    Converts from spherical coordinates to cartesian coordinates
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x,y,z])

def TypeLossNameConverter(type_of_loss):
    """
    Converts from PROPOSAL naming convention of energy losses to PPC ones.
    """
    if(type_of_loss == "epair"):
        return "epair"
    elif(type_of_loss == "brems"):
        return "brems"
    elif(type_of_loss == "photo"):
        return "hadr"
    elif(type_of_loss == "ioniz"):
        return "delta"
    elif(type_of_loss == "continuous"):
        return "delta" # same as ionization, just tracked differently in PROPOSAL
    else:
        raise Exception("Invalid energy loss", type_of_loss)

def power_law_sampler(gamma, xlow, xhig, n, random_state=None):
    r"""
    Sample n events from a power law with index gamma between xlow and xhig
    by using the analytic inversion method. The power law pdf is given by
    .. math::
       \mathrm{pdf}(\gamma) = x^{-\gamma} / \mathrm{norm}
    where norm ensures an area under curve of one. Positive spectral index
    gamma means a falling spectrum.
    Note: When :math:`\gamma=1` the integral is
    .. math::
       \int 1/x \mathrm{d}x = ln(x) + c
    This case is also handled.
    Note: This algorithm was copied from the PROPOSAL examples
    Parameters
    ----------
    gamma : float
        Power law index.
    xlow, xhig : float
        Border of the pdf, needed for proper normalization.
    n : int
        Number of events to be sampled.
    random_state : seed, optional
        Turn seed into a np.random.RandomState instance. See
        `sklearn.utils.check_random_state`. (default: None)
    Returns
    -------
    sample : float array
        Array with the requested n numbers drawn distributed as a power law
        with the given parameters.
    """
    rndgen = check_random_state(random_state)
    # Get uniform random number which is put in the inverse function
    u = rndgen.uniform(size=int(n))

    if gamma == 1:
        return np.exp(u * np.log(xhig / xlow)) * xlow
    else:
        radicant = (u * (xhig**(1. - gamma) - xlow**(1. - gamma)) +
                    xlow**(1. - gamma))
        return radicant**(1. / (1. - gamma))
