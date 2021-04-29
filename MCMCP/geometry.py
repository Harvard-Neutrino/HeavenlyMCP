import numpy as np
import particles as prt
from MCEq.geometry.geometry import EarthGeometry

# Geometry calculator from nuVeto (https://github.com/tianluyuan/nuVeto/blob/master/nuVeto/utils.py)
# slight modification to remove the unnecesary complicated dependencies
class Geometry(EarthGeometry):
    def __init__(self, depth):
        """ Depth of detector and elevation of surface above sea-level
        """
        super(Geometry, self).__init__()
        self.depth = depth
        # changing from "carlos-units" to SI-units
        self.cm = 1.0
        self.meter = 1.0e2*self.cm
        self.h_obs *= self.cm
        self.h_atm *= self.cm
        self.r_E *= self.cm
        self.r_top = self.r_E + self.h_atm
        self.r_obs = self.r_E + self.h_obs

    def overburden(self, cos_theta):
        """Returns the overburden for a detector at *depth* below some surface
        at *elevation*.
        From law of cosines,
        x^2 == r^2+(r-d)^2-2r(r-d)cos(gamma)
        where
        r*cos(gamma) = r-d+x*cos(theta), solve and return x.
        :param cos_theta: cosine of zenith angle in detector coord
        """
        d = self.depth
        r = self.r_E
        z = r-d
        return (np.sqrt(z**2*cos_theta**2+d*(2*r-d))-z*cos_theta)/self.meter


    def overburden_to_cos_theta(self, l):
        """Returns the theta for a given overburden for a detector at 
        *depth* below some surface at *elevation*.
        From law of cosines,
        x^2 == r^2+(r-d)^2-2r(r-d)cos(gamma)
        where
        r*cos(gamma) = r-d+x*cos(theta), solve and return x.
        :param cos_theta: cosine of zenith angle in detector coord
        """
        d = self.depth
        r = self.r_E
        z = r-d
        return (2*d*r-d**2-l**2)/(2*l*z)


    def cos_theta_eff(self, cos_theta):
        """ Returns the effective cos_theta relative the the normal at earth surface.
        :param cos_theta: cosine of zenith angle (detector)
        """
        d = self.depth
        r = self.r_E
        z = r-d
        return np.sqrt(1-(z/r)**2*(1-cos_theta**2))

def EnergyLossRate(E, mcp):
    """
    Returns the energy loss rate assuming standard rock values, valid to ~ 200 MeV.
    See https://inspirehep.net/literature/1245588 for details.

    Args:
        E (double): energy of particle in GeV
        MCP (object): MCP particle object
    Returns:
        dE/dx (double): energy loss in units of GeV/mwe.
    """
    assert(isinstance(mcp,prt.MCP))
    a = 0.223    # GeV/mwe
    b = 0.464e-3 # 1/mwe
    return (a + b*E)*mcp.epsilon**2

def MeanDistance(Einitial, Efinal, mcp):
    """
    Returns the energy loss rate assuming standard rock values, valid to ~ 200 MeV.
    See https://inspirehep.net/literature/1245588 for details.

    Args:
        Einitial (double): energy of particle in GeV
        Efinal (double): energy of particle in GeV
        MCP (object): MCP particle object
    Returns:
        dE/dx (double): energy loss in units of GeV/cmwe.
    """
    assert(isinstance(mcp,prt.MCP))
    a = 0.223*mcp.epsilon**2*1.e-2    # GeV/mwe
    b = 0.464e-3*mcp.epsilon**2*1.e-2 # 1/mwe
    return np.log((a + b*Einitial)/(a + b*Efinal))/b

__standard_rock_density = 2.650 # g/cm^3
__water_density = 1.0 # g/cm^3
__meter_water_equivalent_to_rock_water_equivalent = __water_density/__standard_rock_density
__km = 1e5 # cm
__Earth = Geometry(1*__km)

def FluxAttenuation(Einitial, Efinal, costh, mcp, Earth = __Earth):
    """
    Returns the attenuation of the flux factor.

    Args:
        Einitial (double): energy of particle in GeV
        Efinal (double): energy of particle in GeV
        costh (double): cosine of the zenith angle
        MCP (object): MCP particle object
        Earth (object): Earth geometry object
    Returns:
        AttenuationFactor (double): dimensionless
    """
    mean_overburden = MeanDistance(Einitial,Efinal,mcp)*__meter_water_equivalent_to_rock_water_equivalent
    return np.exp(-Earth.overburden(costh)/mean_overburden)

