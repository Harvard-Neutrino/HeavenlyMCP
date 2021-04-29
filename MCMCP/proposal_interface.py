import proposal as pp
import numpy as np

def make_mcp(epsilon, mass):
    """
    epsilon: dimensionless
    mass: MeV
    """
    lifetime = 1e100
    decay_table = pp.decay.DecayTable()
    decay_table.set_stable()
    mcp_def = pp.particle.ParticleDef('mcp', mass, mass, lifetime, -1.*epsilon, [], decay_table, 0, 0)
    return mcp_def
    
def make_propagator(mcp_def, ecut = 500, vcut = 1e-4, consider_soft_losses = True):
    """
    mcp_def: MCP object
    ecut: energy cut threshold (MeV)
    vcut: relative energy loss cut threshold (dimensionless)
    consider_soft_losses: soft losses will be tracked individually or not
    """
    geometry = pp.geometry.Sphere(pp.Vector3D(), 1.e20, 0.0)
    sector_def = pp.SectorDefinition()
    sector_def.cut_settings = pp.EnergyCutSettings(ecut, vcut)
    sector_def.medium = pp.medium.Ice()
    sector_def.geometry = geometry
    sector_def.scattering_model = pp.scattering.ScatteringModel.NoScattering
    sector_def.crosssection_defs.brems_def.lpm_effect = False
    sector_def.crosssection_defs.epair_def.lpm_effect = False
    sector_def.do_continuous_energy_loss_output = consider_soft_losses

    detector = geometry

    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = "~/.local/share/PROPOSAL/tables"
    interpolation_def.path_to_tables_readonly = "~/.local/share/PROPOSAL/tables"

    prop = pp.Propagator(mcp_def, [sector_def], detector, interpolation_def)
    return prop

def propagate_mcp(mcp_energies, mcp_def, direction = (0,0,-1), position = (0,0,0),
                  propagation_length = 1e5, save_to_file = False, prop = None):
    """
    mcp_energies: MCP energy array in units of MeV
    mcp_def: PROPOSAL mCP definition
    direction: (vx,vy,vz)
    propagation_length: cm
    save_to_file : will save to a file or not
    prop : use a pregenerated propagator object if provided. Else will make one on the fly.
    """
    if prop == None:
        prop = make_propagator(mcp_def)
    
    energy_losses = {}
    energy_losses["continuous"] = []
    energy_losses["epair"] = []
    energy_losses["brems"] = []
    energy_losses["ioniz"] = []
    energy_losses["photo"] = []

    mcp_prop = pp.particle.DynamicData(mcp_def.particle_type)
    mcp_prop.position = pp.Vector3D(*position)
    mcp_prop.direction = pp.Vector3D(*direction)
    mcp_prop.propagated_distance = 0

    for mcp_energy in mcp_energies:
        mcp_prop.energy = mcp_energy
        secondarys = prop.propagate(mcp_prop, propagation_length)

        for sec in secondarys.particles:
            log_sec_energy = np.log10(sec.parent_particle_energy - sec.energy)
            #log_sec_energy = np.log10(sec.energy)

            if sec.type == int(pp.particle.Interaction_Type.ContinuousEnergyLoss):
                energy_losses["continuous"].append([log_sec_energy,np.array([sec.position.x,sec.position.y,sec.position.z])])
            if sec.type == int(pp.particle.Interaction_Type.Epair):
                energy_losses["epair"].append([log_sec_energy,np.array([sec.position.x,sec.position.y,sec.position.z])])
            if sec.type == int(pp.particle.Interaction_Type.Brems):
                energy_losses["brems"].append([log_sec_energy,np.array([sec.position.x,sec.position.y,sec.position.z])])
            if sec.type == int(pp.particle.Interaction_Type.DeltaE):
                energy_losses["ioniz"].append([log_sec_energy,np.array([sec.position.x,sec.position.y,sec.position.z])])
            if sec.type == int(pp.particle.Interaction_Type.NuclInt):
                energy_losses["photo"].append([log_sec_energy,np.array([sec.position.x,sec.position.y,sec.position.z])])

    if save_to_file:
        # =========================================================
        #   Write
        # =========================================================

        out_filename = 'data_sec_dist_{}_{}_Emcp_{}'.format(
                mcp_def.name,
                sector_def.medium.name.lower(),
                Emcp,
                ecut,
                vcut)
        np.savez(
            out_filename,
            continuous=energy_losses["continuous"],
            brems=energy_losses["brems"],
            epair=energy_losses["epair"],
            photo=energy_losses["photo"],
            ioniz=energy_losses["ioniz"],
            statistics=[statistics],
            E_min=[E_min_log],
            E_max=[E_max_log],
            spectral_index=[spectral_index],
            distance=[propagation_length / 100],
            medium_name=[sector_def.medium.name.lower()],
            particle_name=[mcp_def.name],
            ecut=[ecut],
            vcut=[vcut]
        )
        return out_filename + ".npz"
    else:
        energy_losses["continuous"] = np.array(energy_losses["continuous"],dtype='object')
        energy_losses["brems"] = np.array(energy_losses["brems"],dtype='object')
        energy_losses["epair"] = np.array(energy_losses["epair"],dtype='object')
        energy_losses["photo"] = np.array(energy_losses["photo"],dtype='object')
        energy_losses["ioniz"] = np.array(energy_losses["ioniz"],dtype='object')
        return energy_losses

if __name__ == "__main__":
    outfile = propagate_mcp(1e3,0.1, 105.6583745, True) # energy, epsilon, mass in MeV
#
