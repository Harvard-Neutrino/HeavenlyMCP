#--------------------------------------------------------------------------------------------------------------------------------------
#  This script contains useful functions to compute the event rate in a detector taking into account the angular distribution of MCPs
#--------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy import integrate

import os
cwd = os.getcwd() #os.path.dirname(os.getcwd())
#print(cwd)
print('Rate_for_cosine.py called! ')


#--- Load the data and create a data frame
path=cwd+'\Data\MCP_flux'
df_detec=pd.read_csv(path+r'\MCP_Detector_AllcosTh.csv' )
df_det_sort=df_detec.sort_values(by=['cos'])

#--- Define a funtion to get the flux for a fixed mass:
#---
m_tst=0.01 # this is the MCP mass in MeV
print('mass fixed at '+str(m_tst)+' MeV')
#---
def Make_MCP(cosine,eps2):
    cos_v=cosine
    eps2_v=eps2
    df_fil=df_det_sort[df_det_sort['mass'] == m_tst]
    df_fil2=df_fil[df_fil['cos'] == cos_v]
    df_fil3=df_fil2[df_fil2['eps2'] == eps2_v]
    return df_fil3

def MCP_flux(cosine,eps2,E):
    E_v=E
    cos_v=cosine
    eps2_v=eps2
    df=Make_MCP(cos_v,eps2_v)
    interpol=interpolate.interp1d(df['e'],df['flux'])
    return float(interpol(E_v))

#--- Set E_values:
E_val=df_det_sort['e'].unique()
#--- Set cos values:
cos_values=df_det_sort['cos'].unique()

#--- Set relevant constants and unit conversion
GeVM1_cm=(5.06e13)**-1
MeV_to_GeV=1e-3
alpha=137**-1
me=0.000511 # electron mass in GeV
NA = 6.02e23 # Avogadro's Constant
rho_water = 1.00 # density in [gr/cm3]
V=1e9 # one ktnn in [cm3]
ne=rho_water*NA*V # number of electrons in the detector

#--- Define the cross-section and the object to integrate

def Er_max(E_MCP,m_MCP):
    return (E_MCP**2-m_MCP**2)*me/(m_MCP**2+2*E_MCP*me+me**2)

def dsigma_dEr(Er,E,mMCP,eps2):
    # Input in GeV
    # Output in GeV^-3
    Num=2*E**2*me + Er**2*me - Er*(mMCP**2+me*(2*E+me))
    Den=Er**2*(E**2-mMCP**2)*me**2
    return np.pi*alpha**2*eps2*Num*Den**-1

def XS(E,Ermin,eps2):
    # Input in GeV
    # Output in cm^2
    E_mcp=E
    Ermax=Er_max(E_mcp,m_tst)
    eps2_v=eps2
    I=integrate.quad(dsigma_dEr,Ermin,Ermax,args=(E_mcp,m_tst,eps2_v),epsabs=1e-10,epsrel=1e-10,limit=300)
    return I[0]*GeVM1_cm**2

def integrando(E,Ermin,eps2,cosine):
    # Input in GeV
    # Output in (s sr)^-1
    return MCP_flux(cosine,eps2,E)*XS(E,Ermin,eps2)*(2*np.pi)**-1

#--- Define function that computes the event rate vs cos\theta for different values of eps2 and a fixed value of the minimum recoil

def Rate_for_cosine(Erm,eps2_val):
    Ermin=Erm*MeV_to_GeV
    rate_val=[]
    for eps2_v in eps2_val:
        rate_eps2=[]
        cos_eps2=[]
        for cos in df_det_sort['cos'].unique():
            I=integrate.quad(integrando,min(E_val),max(E_val),args=(Ermin,eps2_v,cos),epsabs=1e-10,epsrel=1e-10,limit=300)
            rate_eps2.append(ne*I[0])
        rate_val.append(rate_eps2)
    return rate_val