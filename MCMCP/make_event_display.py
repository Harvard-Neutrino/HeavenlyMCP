import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import tools

"""
Plotting script originally written by J. Lazar and adapted by C. Arg\"uelles for this Monte Carlo.
"""

def PlotIceCubeDetectorDOMLocations(icecube_geometry_f2k):
    """
        Plots the DOM locations according to the provided geometry file.
    """
    # plot the IceCube detector DOMs
    geofile = np.genfromtxt(icecube_geometry_f2k)

    X = geofile[:,2]
    Y = geofile[:,3]
    Z = geofile[:,4]

    fig = plt.figure(figsize=(15,15))
    ax  = fig.add_subplot(111, projection='3d')

    times   = np.linspace(0,100, len(X))
    charges = np.random.rand(len(X))*200

    scat = ax.scatter(X, Y, Z, c=times, alpha=0.5, s=charges)
    cbar = fig.colorbar(scat, shrink=0.5, aspect=5)

    plt.show()

def ReadEventInfo(ppc_hits_file):
    """
        Reads the output from PPC into HIT information and MC information for each event.
        HITINFO contains the list of DOMs that were hit and with how much light.
        LOSSESINFO contains the type of loss, location, and amount.
        MCINFO contains the event true properties: Energy, Interaction Vertex, Zenith Angle, Azimuthal Angle
    """
    hitinfo = []
    lossinfo = []
    mcinfo = []
    with open(ppc_hits_file, 'r') as fp:
        ehitinfo = []
        elossinfo = []
        for line in fp:
            if("HIT" in line):
                x = list(map(float,line.split(" ")[1:]))
                x.insert(0,0)
                x[1] = float(x[1])
                x[2] = float(x[2])
                ehitinfo.append(x)
            if("TR" in line):
                #x = list(map(float,line.split(" ")[1:]))
                x = list(line.split(" "))
                # type of loss, x, y, z, energy_deposited
                elossinfo.append([x[3],float(x[4]),float(x[5]),float(x[6]),float(x[10])])
            if("MC" in line):
                x = list(line.split(" "))
                mcinfo.append(list(map(float,([x[2],x[4],x[6],x[8],x[10],x[12]])))) # E/MeV, x/cm, y/cm, z/cm, theta, phi
                mcinfo[-1][0] *= 1e-3 # convert from MeV to GeV
                mcinfo[-1][1] *= 1e-2 # convert from cm to m
                mcinfo[-1][2] *= 1e-2 # convert from cm to m
                mcinfo[-1][3] *= 1e-2 # convert from cm to m
            if("EE" in line):
                hitinfo.append(np.array(ehitinfo))
                lossinfo.append(np.array(elossinfo))
                ehitinfo = []
                elossinfo = []
    return hitinfo, lossinfo, mcinfo

def MakeEventDictionary(hitinfo, geofile = np.genfromtxt('./PPC/geo-f2k')):
    """
        Reads the output from PPC into HIT information and MC information for each event.
        HITINFO contains the list of DOMs that were hit and with how much light.
        MCINFO contains the event true properties: Energy, Interaction Vertex, Zenith Angle, Azimuthal Angle
    """
    event_dict   = {tuple(tup[:2]):(tuple(tup[2:]),[]) for tup in zip(geofile[:,5],geofile[:,6],geofile[:,2],geofile[:,3],geofile[:,4])
                 }
    for hit in hitinfo:
        nstr = hit[1]
        ndom = hit[2]
        event_dict[(nstr,ndom)][1].append(hit[3])
    return event_dict

def PlotEventHits(event_dict, hitinfo, lossinfo, mcinfo, fig_name,
                  show_array = True,
                  show_mc_track = True,
                  show_losses = True,
                  loss_threshold = 0.1,
                  interactive = True,
                  plot_no_hit_events = False
                 ):
    """
        Plots the given event.
        event_dict: dictionary of hits, where the dictionary key is the DOM index.
        hitinfo: information about each hit
        lossinfo: information about losses from PROPOSAL
        mcinfo: true information of the trajectories
        fig_name: output file name
        show_array: plots the IceCube array on the background
        show_mc_track: plots the true trajectory direction
    """
    if(len(hitinfo) == 0 and not plot_no_hit_events):
        return

    fig = plt.figure(figsize=(15,15))
    ax  = fig.add_subplot(111, projection='3d')

    if(len(hitinfo)>0):
        X = [event_dict[(nstr,ndom)][0][0] for nstr in sorted(list(set(hitinfo[:,1])))
                                           for ndom in sorted(list(set(hitinfo[:,2])))
            ]
        Y = [event_dict[(nstr,ndom)][0][1] for nstr in sorted(list(set(hitinfo[:,1])))
                                           for ndom in sorted(list(set(hitinfo[:,2])))
            ]
        Z = [event_dict[(nstr,ndom)][0][2] for nstr in sorted(list(set(hitinfo[:,1])))
                                           for ndom in sorted(list(set(hitinfo[:,2])))
            ]

        times   = np.array(
                           [np.mean(event_dict[(nstr,ndom)][1]) for nstr in sorted(list(set(hitinfo[:,1])))
                                                                for ndom in sorted(list(set(hitinfo[:,2])))
                           ]
                          ) 
        charges = np.array(
                           [30*(1+np.log(len(event_dict[(nstr,ndom)][1]))) for nstr in sorted(list(set(hitinfo[:,1])))
                                                                           for ndom in sorted(list(set(hitinfo[:,2])))
                           ]
                          )
        scat = ax.scatter(X, Y, Z, c=times, alpha=0.5, s=charges, cmap='hsv', zorder=10)
        cbar = fig.colorbar(scat, shrink=0.5, aspect=5)

    if show_losses:
        z_offset = 1948.07
        markers = {'hadr':'.','epair':'^','delta':'v','brems':'o'}
        for loss in lossinfo:
            loss_type = loss[0]
            position = list(map(float,loss[1:4]))
            amount = float(loss[-1])
            if(amount>loss_threshold):
                ax.scatter(position[0],position[1],position[2]-z_offset,
                           marker = markers[loss_type],
                           s = np.sqrt(amount))

    if show_mc_track:
        z_offset = 1948.07
        #z_offset = 1970.0
        #z_offset = 0.0
        dir = tools.GetCartesianPoint(mcinfo[4],mcinfo[5],-1)
        point = np.array([mcinfo[1],mcinfo[2],mcinfo[3]-z_offset])
        # find point of closest approach to the detector to center the track
        opoint = np.array([0.0,0.0,-1900.0])
        dir_norm = LA.norm(dir)**2
        t_0 = (- dir[0]*(point[0] - opoint[0]) - dir[1]*(point[1] - opoint[1]) - dir[2]*(point[2] - opoint[2]))/dir_norm
        cpoint = point + dir*t_0
        energy = mcinfo[0]
        # make track
        tMC = np.linspace(-700.0,700.0,100)
        x_track = cpoint[0] + dir[0]*tMC
        y_track = cpoint[1] + dir[1]*tMC
        z_track = cpoint[2] + dir[2]*tMC
        #ax.scatter(point[0],point[1],point[2], "*", color = "green", )
        #ax.scatter(opoint[0],opoint[1],opoint[2], "*", color = "red", )
        #ax.scatter(cpoint[0],cpoint[1],cpoint[2], "*", color = "blue", )
        ax.plot(x_track,y_track,z_track, color = "gray", lw = 0.75)
        ax.text2D(0.05, 0.85, "E = {E:0.2f} GeV".format(E = energy), transform = ax.transAxes)

    if show_array:
        AX = [event_dict[key][0][0] for key in event_dict.keys() if key[1]<60] # 60 excludes IceTop3<<
        AY = [event_dict[key][0][1] for key in event_dict.keys() if key[1]<60]
        AZ = [event_dict[key][0][2] for key in event_dict.keys() if key[1]<60]
        scat = ax.scatter(AX, AY, AZ, c='black', alpha=0.2, s=2)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # rotate the axes and update
    ax.view_init(0, 30)
    plt.savefig(fig_name, bbox_inches = "tight")
    if(interactive):
        plt.show()
    plt.clf()
    plt.close()

if __name__ == "__main__":
    import argparse
    import os, sys
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',dest='input',type=str,default="mcp_hits.ppc",
                        help='Input hits in PPC format')
    parser.add_argument('--input_hits',dest='input',type=str,default="mcp_hits.ppc",
                        help='Input hits in PPC format')
    parser.add_argument('-o',dest='output_root',type=str,default="mcp_display",
                        help='Output filename root')
    parser.add_argument('-f',dest='format',type=str,default="pdf",
                        help='Plot output format')
    parser.add_argument('-g',dest='geometry',type=str,default="./PPC/geo-f2k",
                        help='Input detector geometry in F2K format.')
    parser.add_argument('--no-array',dest='array',default=True, action = 'store_false',
                        help='Plots the MC track.')
    parser.add_argument('--mctrack',dest='mctrack',default=False, action = 'store_true',
                        help='Plots the MC track.')
    parser.add_argument('--losses',dest='losses',default=False, action = 'store_true',
                        help='Plots the losses locations from PROPOSAL.')
    parser.add_argument('--losses_threshold',dest='losses_threshold',type=float,default=0.1,
                        help='Minimum loss size to plot.')
    parser.add_argument('--interactive',dest='interactive',default=False, action = 'store_true',
                        help='Minimum loss size to plot.')

    args = parser.parse_args()
    hitinfo, lossinfo, mcinfo = ReadEventInfo(args.input)
    geofile = np.genfromtxt(args.geometry)
    events_dictionaries = [MakeEventDictionary(hits,geofile) for hits in hitinfo]
    print(len(events_dictionaries),np.shape(lossinfo), np.shape(mcinfo))
    [PlotEventHits(events_dictionaries[i],
                   hitinfo[i],
                   lossinfo[i],
                   mcinfo[i],
                   f"./figures/{args.output_root}_{i}.{args.format}",
                   show_array = True,
                   show_mc_track = args.mctrack,
                   show_losses = args.losses,
                   loss_threshold = args.losses_threshold,
                   interactive = args.interactive,
                   plot_no_hit_events = False
                   ) for i in range(len(hitinfo))]
