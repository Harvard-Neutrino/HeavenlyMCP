#---------------------------------------------------------------------------------------------
#       This script can be used to compute the fraction of MCP generated from a Meson decay
#       Details can be found at https://arxiv.org/pdf/1812.03998.pdf
#---------------------------------------------------------------------------------------------
import numpy as np
from math import pi as PI
from scipy import integrate
print('Meson_Br.py called! ')

def integrando_I3(z,x):
    if 1 - (4*x*z**-1)<0 :
        raise ValueError('Error!')
    else:
        return np.sqrt(1 - (4*x*z**-1) )* (1-z)**3 *(2*x+z) * z**-2

def I3(x):
    return (2*(3*PI)**-1)* integrate.quad(integrando_I3,4*x,1,args=x,epsabs= 1e-10, epsrel= 1e-10, limit=300 )[0]

#--- pi0 and eta
alpha_EM = 137**-1
Br_pi0_2gamma=0.98823
m_pi0=0.134976 #GeV
Br_eta_2gamma=0.3941
m_eta=0.548 #GeV

def Br_pi0_mCP(eps2,mMC):
    return 2*eps2*alpha_EM*Br_pi0_2gamma*I3(mMC**2*m_pi0**-2)

def Br_eta_mCP(eps2,mMC):
    return 2*eps2*alpha_EM*Br_eta_2gamma*I3(mMC**2*m_eta**-2)

#---------------------- 2-BODY DECAY

def I2(x,y):
    if 1-4*x<0 or 1-4*y<0:
        raise ValueError('Error!')
    else:
        return (1+2*x)*np.sqrt(1-4*x)* ( (1+2*y)*np.sqrt(1-4*y))**-1
    
#--- Rho, Omega, Phi, JPsi

me= 0.000511 # in (GeV)

m_jpsi=3.1 #  in (GeV)
ctau_jpsi= (7.2e-21)*3e5  # in (km)

m_rho=0.776 #  in  (GeV)
ctau_rho= (4.5e-24)*3e5  # in (km)

m_phi=1.019 #  in  (GeV)
ctau_phi= (1.55e-22)*3e5  # in (km)

m_omega=0.783 # in (GeV)
ctau_omega= (7.75e-23)*3e5 # in (km)

Br_JPsi_2e=5.94/100
Br_Omega_2e=7.28e-5
Br_Phi_2e= 2.954e-4
Br_Rho_2e=4.72e-5

def Br_JPsi_mCP(eps2,mMC):
    return 2*eps2*Br_JPsi_2e*I2(mMC**2*m_jpsi**-2,me**2*m_jpsi**-2) 

def Br_Omega_mCP(eps2,mMC):
    return 2*eps2*Br_Omega_2e*I2(mMC**2*m_omega**-2,me**2*m_omega**-2)    

def Br_Phi_mCP(eps2,mMC):
    return 2*eps2*Br_Phi_2e*I2(mMC**2*m_phi**-2,me**2*m_phi**-2)    

def Br_Rho_mCP(eps2,mMC):
    return 2*eps2*Br_Rho_2e*I2(mMC**2*m_rho**-2,me**2*m_rho**-2)    