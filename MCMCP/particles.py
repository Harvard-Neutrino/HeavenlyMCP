import enum
import numpy as np
from math import pi as PI
from scipy import integrate
from functools import lru_cache

# Original code by V. Mu\~noz
# modified by C.Arg\"uelles

alpha_EM = 137**-1

class Meson(enum.Enum):
    pi0 = 0
    eta = 1
    jpsi = 2

    def beta(self,E):
        m = self.mass()
        if E < m:
            raise ValueError('Error')
        else:
            return np.sqrt(E**2-m**2)/E

    def gamma(self,E):
        return E/self.mass()

    def dec_length(self,E): # in cm!
        return self.gamma(E)*self.beta(E)*self.ctau()

    def dP_dE(self, Eparent, Edaughter, Mdaughter):
        mN = Mdaughter
        EN = Edaughter
        m_p= MesonMass[self]
        gamma_p = Eparent / m_p
        z = EN / gamma_p
        E_max = (mN ** 2 + m_p ** 2)/(2 * m_p) # max neutrino energy in its parent's rest frame
        z_min = E_max - np.sqrt(E_max ** 2 - mN ** 2)
        z_max = E_max + np.sqrt(E_max ** 2 - mN ** 2)
        if (z > z_max) or (z < z_min):
            return 0
        X_p = (- mN**8. +
           8. * mN**6. * m_p ** 2. -
           24. * mN**4. * m_p ** 4. * np.log(mN/m_p) -
           8. * mN**2. * m_p ** 6. +
           m_p ** 8.0) / (24 * m_p)
        bracket = (mN**6. * (4 * m_p ** 2. - 5. * m_p * z - 5. * z ** 2.) -
               9. * mN ** 4. * m_p ** 2. * z * (m_p - 3. * z) +
               9. * mN ** 2. * m_p ** 2. * z ** 3. * (z - 3. * m_p) +
               m_p ** 3. * z ** 3. * (5. * m_p ** 2. + 5 * m_p * z - 4. * z ** 2.))
        invG_dGdz = 1 / (72 * z ** 3. * X_p) * (m_p - z) * bracket
        invG_dGdE = invG_dGdz / gamma_p  
        return invG_dGdE

    def mass(self):
        return MesonMass[self]

    def ctau(self):
        return MesonCTAU[self]

    def latex(self):
        return MesonNameLatex[self]

OnePhotonBranchingFraction = {
        Meson.pi0:0.0000,
        Meson.eta:0.0422,
        Meson.jpsi:0.088
}

TwoPhotonBranchingFraction = {
        Meson.pi0:0.98823,
        Meson.eta:0.3941,
        Meson.jpsi:0.000
}

MesonMass = {
        Meson.pi0:0.134976, # GeV
        Meson.eta:0.547862, # GeV
        Meson.jpsi:3.09600  # GeV
}

MesonCTAU = {
        Meson.pi0:8.4*1e-17 * 3e10, #cm
        Meson.eta:5.0*1e-19* 3e10, #cm
        Meson.jpsi:7.2*1e-20* 3e10 #cm
}

MesonNameLatex = {
    Meson.pi0:r"\pi_0",
    Meson.eta:r"\eta",
    Meson.jpsi:r"J/\psi"
}

class MCP:
    # constructor
    def __init__(self, epsilon, mass):
        self.epsilon = epsilon
        self.mass = mass
    # private methods
    def _integrando_I3(self,z,x):
        return np.sqrt(1 - (4*x*z**-1) )* (1-z)**3 *(2*x+z) * z**-2
    def _I3(self,x):
        return (2*(3*PI)**-1)* integrate.quad(self._integrando_I3,4*x,1,
                                              args=x,
                                              epsabs= 1e-10, epsrel= 1e-10, limit=300)[0]
    # public methods
    @lru_cache(maxsize=2**12)
    def GetBranching(self, meson):
        eps2 = self.epsilon**2
        mCP = self.mass
        mMeson = MesonMass[meson]
        if mCP > mMeson:
            return 0.0
        Br_2gamma = TwoPhotonBranchingFraction[meson]
        return 2*eps2*alpha_EM*Br_2gamma*self._I3(mCP**2*mMeson**-2)

