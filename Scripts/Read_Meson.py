#---------------------------------------------------------------------------------------------
#       This script can be used to read the Meson tables found in the 'Data/Mesons' folder
#       This example is for pi0, for other Mesons you have to change the file name
#---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy import integrate
import os
cwd = os.path.dirname(os.getcwd()) 
#print(cwd)
path=cwd+'\Data\Mesons'
print('Read_Meson.py called! ')

#--- Default meson table is for pi0
cos_d,rho_d,E_d,h_d,dphi_pi0,X_d= np.transpose(np.array(pd.read_csv(path+r'\pi0_file.csv')))

#--- set cosine values that will be used to process the data
cos_unique = np.unique(cos_d)

flx_pi0=[]
flx_eta=[]
flx_rho=[]
flx_phi=[]
flx_omega=[]
dist=[]
slant=[]
energ=[]
heig=[]
dens=[]

for k, co in enumerate(cos_unique):
    array_pi0=[]
    array_eta=[]
    array_rho=[]
    array_phi=[]
    array_omega=[]
    array2=[]
    array3=[]
    array4=[]
    array5=[]
    array6=[]
    for i, cos in enumerate(cos_d):
        if cos_d[i] == cos_unique[k]:
            array_pi0.append(dphi_pi0[i])
            array_eta.append(dphi_pi0[i])
            array_rho.append(dphi_pi0[i])
            array_phi.append(dphi_pi0[i])
            array_omega.append(dphi_pi0[i])
            array2.append(h_d[i])
            array3.append(X_d[i])
            array4.append(E_d[i])
            array5.append(h_d[i])
            array6.append(rho_d[i])
    flx_pi0.append(array_pi0)
    flx_eta.append(array_eta)
    flx_rho.append(array_rho)
    flx_phi.append(array_phi)
    flx_omega.append(array_omega)
    dist.append(array2)
    slant.append(array3)
    energ.append(array4)
    heig.append(array5)
    dens.append(array6)

def FLUJO_diff(string):

    phi_pi0=flx_pi0[string]
    phi_eta=flx_eta[string]
    phi_rho=flx_rho[string]
    phi_phi=flx_phi[string]
    phi_omega=flx_omega[string]
    X0=slant[string]
    E0= energ[string]
    rho0=dens[string]
    l0=dist[string]
    h0=heig[string]
    
    df_l_pi0 = pd.DataFrame.from_dict({'X':l0, 'E':E0, 'F': phi_pi0})
    df_X_pi0 = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi_pi0})

    df_l_eta = pd.DataFrame.from_dict({'X':l0, 'E':E0, 'F': phi_eta})
    df_X_eta = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi_eta})

    df_l_rho = pd.DataFrame.from_dict({'X':l0, 'E':E0, 'F': phi_rho})
    df_X_rho = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi_rho})

    df_l_phi = pd.DataFrame.from_dict({'X':l0, 'E':E0, 'F': phi_phi})
    df_X_phi = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi_phi})

    df_l_omega = pd.DataFrame.from_dict({'X':l0, 'E':E0, 'F': phi_omega})
    df_X_omega = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi_omega})

    def bivariate_interpolation(df, flujo= 'F', slant='X', E='E'): 
        df_copy = df.copy().sort_values(by=[slant, E], ascending=True)
        x = np.array(df_copy[E].unique(), dtype='float64')
        y = np.array(df_copy[slant].unique(), dtype='float64')
        z = np.array(df_copy[flujo].values, dtype='float64')
        Z = z.reshape(len(y), len(x))

        interp_spline = interpolate.RectBivariateSpline(y, x, Z, kx=1, ky=1)
        return interp_spline

    interp_l_pi0= bivariate_interpolation(df_l_pi0)
    interp_X_pi0= bivariate_interpolation(df_X_pi0)
    
    interp_l_eta= bivariate_interpolation(df_l_eta)
    interp_X_eta= bivariate_interpolation(df_X_eta)

    interp_l_rho= bivariate_interpolation(df_l_rho)
    interp_X_rho= bivariate_interpolation(df_X_rho)
    
    interp_l_phi= bivariate_interpolation(df_l_phi)
    interp_X_phi= bivariate_interpolation(df_X_phi)

    interp_l_omega= bivariate_interpolation(df_l_omega)
    interp_X_omega= bivariate_interpolation(df_X_omega)
    
    def dFlux_l_pi0(l,e):
        return interp_l_pi0(l, e, grid=False) 
    def dFlux_X_pi0(x,e):
        return  interp_X_pi0(x, e, grid=False) 

    def dFlux_l_eta(l,e):
        return interp_l_eta(l, e, grid=False) 
    def dFlux_X_eta(x,e):
        return  interp_X_eta(x, e, grid=False) 

    def dFlux_l_rho(l,e):
        return interp_l_rho(l, e, grid=False) 
    def dFlux_X_rho(x,e):
        return  interp_X_rho(x, e, grid=False) 
    
    def dFlux_l_phi(l,e):
        return interp_l_phi(l, e, grid=False) 
    def dFlux_X_phi(x,e):
        return  interp_X_phi(x, e, grid=False) 

    def dFlux_l_omega(l,e):
        return interp_l_omega(l, e, grid=False) 
    def dFlux_X_omega(x,e):
        return  interp_X_omega(x, e, grid=False) 

    return(dFlux_l_pi0,dFlux_X_pi0,dFlux_l_eta,dFlux_X_eta,dFlux_l_rho,dFlux_X_rho,dFlux_l_phi,dFlux_X_phi,dFlux_l_omega,dFlux_X_omega)

arr_flx_l_pi0_corr = []
arr_pi0_X = []
arr_flx_l_eta_corr = []
arr_flx_X_eta_corr = []
arr_flx_l_rho_corr = []
arr_flx_X_rho_corr = []
arr_flx_l_phi_corr = []
arr_flx_X_phi_corr = []
arr_flx_l_omega_corr = []
arr_flx_X_omega_corr = []
arr_h_X=[]
arr_l_X=[]
arr_rho_X=[]
arr_rho_l=[]
arr_h_l=[]
arr_X_l=[]

for ic, cos in enumerate(cos_unique):
    arr_flx_l_pi0_corr.append(FLUJO_diff(ic)[0])
    arr_pi0_X.append(FLUJO_diff(ic)[1])
    arr_flx_l_eta_corr.append(FLUJO_diff(ic)[2])
    arr_flx_X_eta_corr.append(FLUJO_diff(ic)[3])
    arr_flx_l_rho_corr.append(FLUJO_diff(ic)[4])
    arr_flx_X_rho_corr.append(FLUJO_diff(ic)[5])
    arr_flx_l_phi_corr.append(FLUJO_diff(ic)[6])
    arr_flx_X_phi_corr.append(FLUJO_diff(ic)[7])
    arr_flx_l_omega_corr.append(FLUJO_diff(ic)[8])
    arr_flx_X_omega_corr.append(FLUJO_diff(ic)[9])
    
    int_h_X=interpolate.interp1d(slant[ic], heig[ic])
    arr_h_X.append(int_h_X)
    int_rho_X=interpolate.interp1d(slant[ic], dens[ic])
    arr_rho_X.append(int_rho_X)
    int_l_X=interpolate.interp1d(slant[ic], dist[ic])
    arr_l_X.append( int_l_X)
    int_rho_l=interpolate.interp1d(dist[ic], dens[ic])
    arr_rho_l.append(int_rho_l)
    int_h_l=interpolate.interp1d(dist[ic], heig[ic])
    arr_h_l.append(int_h_l) 
    int_X_l=interpolate.interp1d(dist[ic], slant[ic])
    arr_X_l.append(int_X_l) 