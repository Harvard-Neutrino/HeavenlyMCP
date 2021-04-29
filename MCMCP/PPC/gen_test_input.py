import numpy as np

for i in range(100):
    x=(2*np.random.random()-1)*500 # meters
    y=(2*np.random.random()-1)*500 # meters
    z=(2*np.random.random()-1)*500 # meters
    zenith=np.random.random()*180  # degrees
    azimuth=np.random.random()*360 # degrees
    l=500              # length, m
    energy=1.e5        # GeV
    t=0                # ns

    print("EM 1 1 1970 0 0 0")
    print("TR 1 0 e    ", x, y, z, zenith, azimuth, 0, energy, t)
    print("TR 1 0 amu  ", x, y, z, zenith, azimuth, l, energy, t)
    print("TR 1 0 hadr ", x, y, z, zenith, azimuth, 0, energy, t)
    print("EE")
