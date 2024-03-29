## About this repository 

This repo contains code and scripts to reproduce the data for the figures in the applications section of the article [Maxian et al. "A fast spectral method for electrostatics in doubly periodic slit channels"](https://arxiv.org/abs/2101.07088).  


See README inside each folder for information about each simulation  

The doubly periodic Poisson solver is provided by [UAMMD](https://github.com/RaulPPelaez/uammd), which is included as a submodule inside the source folder. **Make sure to clone this repo recursively.**  
This means that you must clone using  
```shell
git clone --recursive https://github.com/stochasticHydroTools/DPPoissonTests
```
source/PoissonSlab.cu encodes a generic Brownian Dynamics simulation of charges that interact electrostatically via this doubly periodic solver. See the file for specific usage instructions. This code allows to also run triply periodic electrostatics via a fast GPU spectral Ewald method also provided by UAMMD.   

See the related UAMMD wiki pages for [triply periodic](https://github.com/RaulPPelaez/UAMMD/wiki/SpectralEwaldPoisson) and [doubly periodic](https://github.com/RaulPPelaez/UAMMD/wiki/DPPoisson) electrostatics for additional information about those modules.  


## USAGE:  

Run make to compile, you might have to adapt it to your particular system before.  
Running "$make test" will go into each test folder and run test.bash  

Parameters are provided via the data.main files inside each folder. You may adapt/modify this file to simulate different systems.   
You can see a list of the available options down below, additional information is available in source/PoissonSlab.cu and each folder's README.md  

### data.main options

The executable created by the Makefile expects to find a file called data.main in the folder where it is executed. Alternatively the data.main file can be specify as its first argument.  

data.main contains a series of parameters that allow to customize the simulation and must have the following  format:  
```shell  
#Lines starting with # are ignored  
option [argument] #Anything after the argument is ignored  
flag #An option that does not need arguments  
```

The following options are available:  
  
**numberParticles** The number of charges in the simulation  
**gw** The Gaussian width of the charges  

**H** The width of the domain  
**Lxy** The dimensions of the box in XY  
**permitivity** Permittivity inside the slab  
**permitivityBottom** Below z=-H/2. If the value is negative it means metallic boundary (infinite permittivity).  
**permitivityTop** Above z=H/2. If the value is negative it means metallic boundary (infinite permittivity).  
**bottomWallSurfaceValue** The zero mode value of the Fourier transform of the bottom wall surface value (potential when the boundary is metallic and surface charge otherwise).  
**printDPPoissonFarFieldZeroModeFile** : If present the zero mode of the solution (for Ex, Ey, Ez and phi) in Fourier space for the Far Field in DPPoisson will be printed every printSteps to the provided fileName.  

**temperature** Temperature for the Brownian Dynamics integrator, the diffusion coefficient will be D=T/(6*pi*viscosity*hydrodynamicRadius). This temperature is therefore given in units of energy.  
**viscosity** For BD  
**hydrodynamicRadius** For BD  
**dt** Time step for the BD integrator  

**U0, sigma, r_m, p** Parameters for the ion-ion repulsive interaction. If U0=0 the steric repulsion is turned off.   

**wall_U0, wall_sigma, wall_r_m, wall_p** Parameters for the ion-wall repulsive interaction.   
**imageDistanceMultiplier** Multiplies the distance of the particles to the wall by this amount. For instance, if 2, particles interact with their images, if 1, particles are repelled to the wall (as if the image was at the wall's height).  
**noWall** Optional, if this flag is present particles will not be repelled by the wall.  

**numberSteps** The simulation will run for this many steps  
**printSteps** If greater than 0, the positions and forces will be printed every printSteps steps  
**relaxSteps** The simulation will run without printing for this many steps.  

**outfile** Positions and charge will be written to this file, each snapshot is separated by a #, each line will contain X Y Z Charge. Can be /dev/stdout to print to screen.  
**forcefile** Optional, if present forces acting on particles will written to this file.  
**fieldfile** Optional, if present electric field acting on particles will written to this file.  
**readFile** Optional, if present charge positions will be read from this file with the format X Y Z Charge. numberParticles lines will be read. Can be /dev/stdin to read from pipe.  

**triplyPeriodic** Optional, if this flag is present electrostatics will be solved with a triply periodic spectral ewald solver. Notice that many parameters are not needed in this mode and will be ignored.  

**split** The Ewald splitting parameter. It is mandatory if triply periodic mode is enabled.  
**Nxy** The number of cells in XY. If this option is present split must NOT be present, it will be computed from this. Nxy can be provided instead of split for doubly periodic mode.  

The following accuracy options are optional, the defaults provide a tolerance of 5e-4  
**support** Number of support cells for the interpolation kernel. Default is 10.  
**numberStandardDeviations** Gaussian truncation. Default is 4  
**tolerance** In doubly periodic, determines the cut off for the near field section of the algortihm. In triply periodic mode this represents the overall accuracy of the solver. Default is 1e-4.   
**upsampling** The relation between the grid cell size and gt=sqrt(gw^2+1/(4*split^2)). h_xy= gt/upsampling. default is 1.2  
**useMobilityFromFile**: Optional, if this option is present, the mobility will depend on the height of the particle according to the data in this file. This file must have two columns with a list of normalized heights (so Z must go from -1 to 1) and normalized mobilities (i.e. 6*pi*eta*a*M0) in X, Y and Z. The values for each particle will be spline interpolated from the data provided in the file. The order of the values does not matter. The following example is equivalent to this option not being present at all:  
```bash
--- mobility.dat---
-1.0 1.0  1.0 1.0
 0.0 1.0  1.0 1.0
 1.0 1.0  1.0 1.0
-------------------
```  
**BrownianUpdateRule**: Optional. Can either be EulerMaruyama (default) or Leimkuhler.  
 **idealParticles**: Optional. If this flag is present particles will not interact between them in any way.  
  
### Python interface

Additionally, a python interface is provided that allows to compute the electric field acting on a group of charges using the new solver.  
This interface requires [pybind11](https://github.com/pybind/pybind11) to compile, which is included as a submodule and will be automatically downloaded if this repo is cloned recursively (see "About this repo" above).  
In order to use it you must compile the python wrappers using make (doing ```make python``` or ```make all``` here will also compile the python library).  
A file called uammd.*.so will be created and then "import uammd" can be used inside python. Notice that you might have to customize python\_interface/Makefile for your particular system.  
See python_interface/dppoisson.py for a usage example.  
Once compiled you can use "help(uammd)" for additional usage information.  
