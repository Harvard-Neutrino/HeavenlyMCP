#---------------------------------------------------------------------------------------------
#     This script contains useful functions used to compute the flux of MCP from Meson decays
#---------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy import integrate
print('Meson_Functions.py called! ')

#---- mass and lifetife of mesons 
m_pi0=0.13497 # (GeV)
ctau_pi0= (8.4e-17)*3e10 # (cm)

m_eta=0.54785 # (GeV)
ctau_eta= (5e-19)*3e10  # (cm)

m_jpsi=3.1 # (GeV)
ctau_jpsi= (7.2e-21)*3e10  # (cm)

m_rho=0.776 # (GeV)
ctau_rho= (4.5e-24)*3e10  # (cm)

m_phi=1.019 # (GeV)
ctau_phi= (1.55e-22)*3e10  # (cm)

m_omega=0.783 # (GeV)
ctau_omega= (7.75e-23)*3e10  # (cm)

def beta(E,m):
    if E < m:
        raise ValueError('Error!')
    else:
        return np.sqrt(E**2-m**2)/E
    
def gamma(E,m):
    return E/m

def decay_length_pi0(E): # cm
    return gamma(E,m_pi0)*beta(E,m_pi0)*ctau_pi0 

def decay_length_eta(E): # cm
    return gamma(E,m_eta)*beta(E,m_eta)*ctau_eta 

def decay_length_jpsi(E): # cm
    return gamma(E,m_jpsi)*beta(E,m_jpsi)*ctau_jpsi 

def decay_length_rho(E): # cm
    return gamma(E,m_rho)*beta(E,m_rho)*ctau_rho 

def decay_length_phi(E): # cm
    return gamma(E,m_phi)*beta(E,m_phi)*ctau_phi 

def decay_length_omega(E): # cm
    return gamma(E,m_omega)*beta(E,m_omega)*ctau_omega 

#--- 2D interpolation function:	
def interpolation_2D(df, flujo= 'F_X1X2', slant='X1', E='X2'): 
    df_copy = df.copy().sort_values(by=[slant, E], ascending=True)
    x = np.array(df_copy[E].unique(), dtype='float64')
    y = np.array(df_copy[slant].unique(), dtype='float64')
    z = np.array(df_copy[flujo].values, dtype='float64')
    Z = z.reshape(len(y), len(x))
    interp_spline_D = interpolate.RectBivariateSpline(y, x, Z, kx=1, ky=1)
    return interp_spline_D
#-------------------------- 3-BODY ENERGY DISTRIBUTION --------------------------#

def dP_dE_3B(Eparent, Edaughter, Mdaughter,Mparent):
    mN = Mdaughter
    EN = Edaughter
    m_p=Mparent	
    gamma_p = Eparent / m_p
    z = EN / gamma_p
    E_max = (mN ** 2 + m_p ** 2)/(2 * m_p) 
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

#--- Ep Integral limit:
def Emax(mdau,mpa):  
    return (mdau**2 + mpa**2) / (2*mpa)

def Epar_min(EN,mdau,mpa): 
    if Emax(mdau,mpa)<mdau:
        raise ValueError('Error')
    else:
        return ( EN*mpa) / (Emax(mdau,mpa) + np.sqrt( Emax(mdau,mpa)**2 - mdau**2 ) )

def Epar_max(EN,mdau,mpa): 
    if Emax(mdau,mpa)<mdau:
        raise ValueError('Error')
    else:
        return ( EN*mpa) / (Emax(mdau,mpa) - np.sqrt( Emax(mdau,mpa)**2 - mdau**2 ) )

#-------------------------- 2-BODY ENERGY DISTRIBUTION --------------------------#

def lambdaFunc(mp,md1,md2): # no dim
    return  np.sqrt(1 + (md1**4)/(mp**4) + (md2**4)/(mp**4) - (2*md1**2)/(mp**2) - (2*md2**2)/(mp**2) - (2*md2**2*md1**2)/(mp**4) ) 

def sqrMom_P(E,mp): # GeV
    if E<mp:
        raise ValueError('Error!')
    else: 
         return np.sqrt( (E**2 - mp**2) )

def dP_dE_2B(Ep,mp,md1,md2):
    return 1/(sqrMom_P(Ep,mp)*lambdaFunc(mp,md1,md2)) 

#--- Ep Integral limit:
def lim_sup_Ep(mM,mN,ml,EN):
    return ( 1/(2 *mN**2) )*(EN *(-ml**2+mM**2+mN**2)+np.sqrt( (EN-mN)*(EN+mN)*(-ml-mM+mN)*(ml-mM+mN)*(-ml+mM+mN)*(ml+mM+mN)) )

def lim_inf_Ep(mM,mN,ml,EN):
    return ( 1/(2 *mN**2) )*(EN *(-ml**2+mM**2+mN**2)-np.sqrt( (EN-mN)*(EN+mN)*(-ml-mM+mN)*(ml-mM+mN)*(-ml+mM+mN)*(ml+mM+mN)) )
