import numpy as np
import uammd
perm = permitivity=uammd.Permittivity(top=1.0, inside=1.0, bottom=1.0)
Lxy = 185
H=50
par = uammd.DPPoissonParameters(Lxy=Lxy, H=H, gw = 0.25, permittivity=perm, Nxy=72)
print(par)
numberParticles = 20000

dppoisson = uammd.DPPoisson(par, numberParticles)

xpos=(np.random.rand(numberParticles)-0.5)*Lxy
ypos=(np.random.rand(numberParticles)-0.5)*Lxy
zpos=(np.random.rand(numberParticles)-0.5)*(H-4*par.gw)
positions = np.ravel((xpos, ypos, zpos), order='F')
charges = np.array(np.ones(numberParticles), np.float32)
charges[::2] *= -1;
totalCharge=np.sum(charges)
print("Total system charge is: ", totalCharge)
if totalCharge != 0:
    print("Extra charge will be placed on the walls with opposite sign")

forces = np.array(np.zeros(3*numberParticles), np.float32)

dppoisson.computeForce(positions, charges, forces)

print("Forces: ")
print(forces)
                
