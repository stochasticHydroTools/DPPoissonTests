### ABOUT THIS SIMULATION:

A group of equally charged particles with random sign (but ensuring electroneutrality) are let to evolve in a triply periodic environment via Brownian Dynamics.  
Particles repell each other via a LJ-like potential.  
See testRDF.bash and PoissonSlab.cu for more information.  

### USAGE:

Ensure PoissonSlab.cu is compiled and available in the father directory "../" as "poisson".  
This test requires [GNU datamash](https://www.gnu.org/software/datamash/) to be available in the system.  

Run testRDF.bash  

Several parameters can be customized in this script.  

This script will generate starting positions using tools/init.sh.  
Then run the simulation according to the parameters in the header of testRDF.bash.  
It will then compute and average the radial distribution function (rdf) of several equilibrium configurations along with DHO theoretical results.  
Error bars are given by standard deviation of different realizations of the same simulation.   
Notice that the only rdf stored is the cross rdf between + and - charges (pm).   

Results will be placed under a results folder.  
If the time step is low enough, the contents of rdf.dt*.dat should be similar to rdf.pm.lj.theo and or rdf.pm.hs.theo (which uses hard spheres instead of the LJ-like potential).  

