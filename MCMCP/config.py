import numpy as np
from constants import *

mCPMCSpecifications = {}
mCPMCSpecifications["DetectorCenterDepth"] = 1.94807 # km
mCPMCSpecifications["DetectorRadius"] = 1.0*km_to_cm
mCPMCSpecifications["PropagationPadding"] = 1.0*km_to_cm
mCPMCSpecifications["costh_min"] = 0.0
mCPMCSpecifications["costh_max"] = 1.0
mCPMCSpecifications["phi_min"] = 0.0
mCPMCSpecifications["phi_max"] = 2.*np.pi
mCPMCSpecifications["E_min"] = 0.1*GeV_to_MeV
mCPMCSpecifications["E_max"] = 1000.0*GeV_to_MeV
mCPMCSpecifications["E_gamma"] = 3
mCPMCSpecifications["Energy_sampling"] = "powerlaw"
mCPMCSpecifications["N_samples"] = int(1e3)
mCPMCSpecifications["keep_empty"] = False
mCPMCSpecifications["consider_soft_losses"] = False
mCPMCSpecifications["random_number_seed"] = 0

def CheckConfigDict(MC_specifications):
    for key in mCPMCSpecifications.keys():
        if not key in MC_specifications.keys():
            raise Expection("Missing key in configuration", key)