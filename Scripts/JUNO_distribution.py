#-------------------------------------------------------------------------------------------
#  This script contains useful functions to compute the binned event distribution at JUNO 
#-------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import scipy.stats as stats

import os
cwd = os.getcwd() # os.path.dirname(os.getcwd())
print('JUNO_distribution.py called! ')
#print(cwd)

#--- Load signal data
path_to_signal=cwd+'\Data\Signals'
predictions = np.genfromtxt(path_to_signal+r"\juno_evts_final.csv", delimiter = ",", names = True)
epsilon_range = np.unique(predictions["eps2"])
mass_range = np.unique(predictions["m"])

def GetPrediction(ei, mi):
    return np.array([p for p in predictions if p[0] == epsilon_range[ei] and p[1] == mass_range[mi]])

def GetSumEvts(ei, mi):
    return np.sum([p[1] for p in predictions if p[0] == epsilon_range[ei] and p[1] == mass_range[mi]])

def GetIndexes(eps,mass):
    for i in range(len(epsilon_range)):
        if epsilon_range[i] == eps:
            break
    for j in range(len(mass_range)):
        if mass_range[j] == mass:
            break
    return (i,j)

def GetBinnedPrediction(ei, mi, bin_edges):
    pred = GetPrediction(ei,mi)
    pred_inter = interpolate.interp1d(pred['Energy'],pred['Events'], bounds_error = False, fill_value = 0.0)
    return np.array([integrate.quad(pred_inter,bin_edges[ib], bin_edges[ib+1])[0]/(bin_edges[ib+1]-bin_edges[ib]) for ib in range(len(bin_edges)-1)])

#--- Load background data
path_to_data=cwd+'\Data\Experiments'
background_components = ['B8', 'Be11', 'C10', 'C11', 'NC']
background_table = {}
background_spline = {}
background_binned = {}

SolarBackGroundsFactor = 170.0*365.0*10
bin_edges = np.arange(0,35,1)
bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2.

for background in background_components:
    background_table[background] = np.genfromtxt(path_to_data +r"\{}.csv".format(background), delimiter = "," )
    norm_factor = SolarBackGroundsFactor
    if background == 'NC':
        norm_factor = 1.
    background_spline[background] = interpolate.interp1d(background_table[background][:,0],
                                                         background_table[background][:,1]*norm_factor,
                                                         bounds_error = False,
                                                         fill_value = 0.0)
    background_binned[background] = np.array([integrate.quad(background_spline[background],
                                                 bin_edges[ib],
                                                 bin_edges[ib+1])[0]/(bin_edges[ib+1]-bin_edges[ib])
                                  for ib in range(len(bin_edges)-1)])
