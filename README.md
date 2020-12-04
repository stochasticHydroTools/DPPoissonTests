## About this repository 

This repo contains code and script to reproduce the data for the figures X (folder charged_wall) and Y (folder uncharged_wall) in the article [CITE THE ARTICLE HERE].  

See README inside each folder for information about each simulation  

The doubly periodic Poisson solver is provided by [UAMMD](https://github.com/RaulPPelaez/uammd), which is included as a submodule inside the source folder. **Make sure to clone this repo recursively.**  
source/PoissonSlab.cu encodes a generic Brownian Dynamics simulation of charges that interact electrostatically via this doubly periodic solver. See the file for specific usage instructions.  

## USAGE:  

Run make to compile, you might have to adapt it to your particular system before.  
Running "$make test" will go into each test folder and run test.bash  

Parameters are provided via the data.main files inside each folder. You may adapt/modify this file to simulate different systems.   
You can see a list of the available options down below, additional information is available in source/PoissonSlab.cu  

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
**permitivityBottom** Below z=-H/2  
**permitivityTop** Above z=H/2  

**temperature** Temperature for the Brownian Dynamics integrator, the diffusion coefficient will be D=T/(6*pi*viscosity*hydrodynamicRadius)  
**viscosity** For BD  
**hydrodynamicRadius** For BD  
**dt** Time step for the BD integrator  

**U0, sigma, r_m, p** Parameters for the repulsive interaction. If U0=0 the steric repulsion is turned off.   

**numberSteps** The simulation will run for this many steps  
**printSteps** If greater than 0, the positions and forces will be printed every printSteps steps  
**relaxSteps** The simulation will run without printing for this many steps.  

**outfile** Positions and charge will be written to this file, each snapshot is separated by a #, each line will contain X Y Z Charge. Can be /dev/stdout to print to screen.  
**forcefile** Optional, if present forces acting on particles will written to this file.  
**readFile** Optional, if present charge positions will be read from this file with the format X Y Z Charge. numberParticles lines will be read. Can be /dev/stdin to read from pipe.  

**noWall** Optional, if this flag is present particles will not be repelled by the wall.  

**split** The Ewald splitting parameter  
**Nxy** The number of cells in XY. If this option is present split must NOT be present, it will be computed from this.  

The following accuracy options are optional, the defaults provide a tolerance of 5e-4
**support** Number of support cells for the interpolation kernel. Default is 10.  
**numberStandardDeviations** Gaussian truncation. Default is 4  
**tolerance** Determines the cut off for the near field section of the algortihm. Default is 1e-4  
**upsampling** The relation between the grid cell size and gt=sqrt(gw^2+1/(4*split^2)). h_xy= gt/upsampling. default is 1.2  
  
