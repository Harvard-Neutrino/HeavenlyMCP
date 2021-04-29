import numpy as np
import random
import tqdm
from numpy import linalg as LA
# library modules
import config
import geometry
import tools
import proposal_interface as mcp_gun
from constants import *

def GenerateMCPEvents(MCP_epsilon, MCP_mass, mCPMCSpecifications): 
    """
    epsilon: dimensionless
    mass: MeV
    mCPMCSpecifications: dictionary with MC settings
    """
    # check inputs:
    config.CheckConfigDict(mCPMCSpecifications)
    if MCP_epsilon <= 0 or MCP_mass <= 0:
        raise Exception("Provided MCP mass or epsilon are negative or zero. That is not allowed.")
    
    # make empty array where events will be placed
    mCPEnergyLosses = []
    
    mCPDefinition = mcp_gun.make_mcp(MCP_epsilon,MCP_mass)
    mCPpropagator = mcp_gun.make_propagator(mCPDefinition,
                                            consider_soft_losses = mCPMCSpecifications["consider_soft_losses"])
    
    keep_empty = mCPMCSpecifications["keep_empty"]
    DetectorEarth = geometry.Geometry(mCPMCSpecifications["DetectorCenterDepth"]*geometry.__km)
    
    low_range = [mCPMCSpecifications["costh_min"],mCPMCSpecifications["phi_min"],0.0,mCPMCSpecifications["E_min"]]
    high_range = [mCPMCSpecifications["costh_max"],mCPMCSpecifications["phi_max"],1.0,mCPMCSpecifications["E_max"]]
    
    if mCPMCSpecifications["Energy_sampling"] == "powerlaw":
        E_mCP_array = tools.power_law_sampler(gamma = mCPMCSpecifications["E_gamma"],
                                                xlow = mCPMCSpecifications["E_min"],
                                                xhig = mCPMCSpecifications["E_max"],
                                                n = mCPMCSpecifications["N_samples"],
                                                random_state = mCPMCSpecifications["random_number_seed"])
    
    for ii, (costh, phi, u, E_mCP) in tqdm.tqdm(enumerate([np.random.uniform(low = low_range, high = high_range)
                                      for i in range(mCPMCSpecifications["N_samples"])])):
        # override the energy distribution in the case that the sampling is powerlaw like
        if mCPMCSpecifications["Energy_sampling"] == "powerlaw":
            E_mCP = E_mCP_array[ii]
        
        # compute injection direction
        r = DetectorEarth.overburden(costh)*meter_to_cm
        iposition = tools.GetCartesianPoint(np.arccos(costh),phi,r)
        direction = tools.GetCartesianPoint(np.arccos(costh),phi,-1)
        # compute orthogonal direction
        orthogonal_direction = np.random.randn(3)
        orthogonal_direction -= orthogonal_direction.dot(direction) * direction
        orthogonal_direction /= np.linalg.norm(orthogonal_direction)
        orthogonal_direction *= mCPMCSpecifications["DetectorRadius"]*2.0*u
        # injection vertex
        iposition+=orthogonal_direction
        # send things to PROPOSAL
        losses = mcp_gun.propagate_mcp([E_mCP+MCP_mass],mCPDefinition,
                                       position = iposition, direction = direction, propagation_length= r + mCPMCSpecifications["PropagationPadding"],
                                       prop = mCPpropagator
                                      )
        if (mCPMCSpecifications["consider_soft_losses"]):
            total_loss = np.sum([len(losses[loss]) for loss in losses])
        else: 
            total_loss = np.sum([len(losses[loss]) for loss in losses if loss != "continuous"])
        if(not keep_empty and total_loss == 0):
            continue
        mCPEnergyLosses.append([costh,phi,E_mCP,iposition,losses])
    return mCPEnergyLosses

if __name__ == "__main__":
    import argparse
    import os, sys
    import subprocess

    parser = argparse.ArgumentParser()
    # Output filename
    parser.add_argument('-o',dest='output',type=str,default="mcp_events.f2k",
                        help='Output hits in F2K format.')
    # MCP parameters
    parser.add_argument('--MCP_mass',dest='MCP_mass',type=float,default=100.0,
                        help='Mass of the MCP in MeV.')
    parser.add_argument('--MCP_epsilon',dest='MCP_epsilon',type=float,default=1.0,
                        help='Dimensionless fractional charge value.')
    # Monte Carlo settings
    parser.add_argument('--E_min',dest='E_min',type=float,default=1e2,
                        help='Mininum energy in MeV.')
    parser.add_argument('--E_max',dest='E_max',type=float,default=1e5,
                        help='Maximum energy in MeV.') 
    parser.add_argument('--costh_min',dest='costh_min',type=float,default=0.0,
                        help='Minimun cosine of zenith angle considered.')
    parser.add_argument('--costh_max',dest='costh_max',type=float,default=1.0,
                        help='Maximum cosine of zenith angle considered.')
    parser.add_argument('--phi_min',dest='phi_min',type=float,default=0.0,
                        help='Minimun cosine of zenith angle considered.')
    parser.add_argument('--phi_max',dest='phi_max',type=float,default=2.0*np.pi,
                        help='Maximum cosine of zenith angle considered.')   
    parser.add_argument('--random_number_seed',dest='random_number_seed',type=int,default=0,
                        help='Random number generators seed.')      
    parser.add_argument('--consider_soft_losses',dest='consider_soft_losses',type=bool,default=False,
                        help='PROPOSAL will also print out the soft losses.')      
    parser.add_argument('--N_samples',dest='N_samples',type=int,default=1000,
                        help='Number of events to generate.')      
    parser.add_argument('--keep_empty',dest='keep_empty',type=bool,default=False,
                        help='Keep events that generate no energy losses.')      
    parser.add_argument('--Energy_sampling',dest='Energy_sampling',type=str,default="powerlaw",
                        help='Energy sampling can be: powerlaw or uniform.')      
    parser.add_argument('--E_gamma',dest='E_gamma',type=float,default=3,
                        help='If using powerlaw energy sampling, this is the power law negative exponent.')      
    parser.add_argument('--DetectorCenterDepth',dest='DetectorCenterDepth',type=float,default=1.94807,
                        help='Depth of the detector in kilometers.')      
    parser.add_argument('--DetectorRadius',dest='DetectorRadius',type=float,default=1,
                        help='Detector radius in kilometers.')      
    parser.add_argument('--PropagationPadding',dest='PropagationPadding',type=float,default=1,
                        help='Additional padding length considered after the particle has crosssed the detector.')      
    args = parser.parse_args()
    
    MCPMCSpecifications = config.mCPMCSpecifications
    MCPMCSpecifications["DetectorCenterDepth"] = args.DetectorCenterDepth
    MCPMCSpecifications["DetectorRadius"] = args.DetectorRadius*km_to_cm
    MCPMCSpecifications["PropagationPadding"] = args.PropagationPadding*km_to_cm
    MCPMCSpecifications["costh_min"] = args.costh_min
    MCPMCSpecifications["costh_max"] = args.costh_max
    MCPMCSpecifications["phi_min"] = args.phi_min
    MCPMCSpecifications["phi_max"] = args.phi_max
    MCPMCSpecifications["E_min"] = args.E_min
    MCPMCSpecifications["E_max"] = args.E_max
    MCPMCSpecifications["E_gamma"] = args.E_gamma
    MCPMCSpecifications["Energy_sampling"] = args.Energy_sampling
    MCPMCSpecifications["N_samples"] = args.N_samples
    MCPMCSpecifications["keep_empty"] = args.keep_empty
    MCPMCSpecifications["consider_soft_losses"] = args.consider_soft_losses
    MCPMCSpecifications["random_number_seed"] = args.random_number_seed
    
    MCPEvents = GenerateMCPEvents(args.MCP_epsilon, args.MCP_mass, MCPMCSpecifications)
    tools.WriteToF2KFormat(MCPEvents, args.output)