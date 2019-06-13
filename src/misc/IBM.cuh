/*Raul P. Pelaez 2019. Immersed Boundary Method (IBM).
This class contains functions to spread marker information to a grid and interpolate information from a grid to some marker positions, according to the Immersed Boundary Method [1].
It allows to employ any Kernel to do so, see IBM_kernels.cuh.*-

USAGE:

//Creation, needs the System shared_ptr
using Kernel = IBM_kernels::PeskinKernel::threePoint;
auto ibm = std::make_shared<IBM<Kernel>>(sys, kernel);

//Spread to a grid

ibm->spread(pos,      //An iterator with the position of the markers
            quantity, //An iterator with the quantity of each marker to spread
	    gridQuantity, //A real3* with the grid data (ibm will sum to the existing data)
	    grid,         //A Grid descriptor corresponding to gridQuantity
	    numberMarkers,
	    cudaStream);

//Interpolate from a grid

ibm->gather(pos,      //An iterator with the position of the markers
            quantity, //An iterator with the quantity of each marker to gather (ibm will sum to the existing values)
	    gridQuantity, //A real3* with the grid data
	    grid,         //A Grid descriptor corresponding to gridQuantity
	    numberMarkers,
	    cudaStream);


//Get a reference to the kernel
auto kernel = ibm->getKernel();

REFERENCES:
[1] Charles S. Peskin. The immersed boundary method (2002). DOI: 10.1017/S0962492902000077
*/
#ifndef MISC_IBM_CUH
#define MISC_IBM_CUH
#include"global/defines.h"
#include"utils/utils.h"
#include"System/System.h"
namespace uammd{
  template<class Kernel>
  class IBM{
    shared_ptr<Kernel> kernel;
    shared_ptr<System> sys;
  public:

    IBM(shared_ptr<System> sys, shared_ptr<Kernel> kern);

    template<class PosIterator, class QuantityIterator>
    void spread(const PosIterator &pos, const QuantityIterator &v,
		real3 *gridData,
		Grid grid, int numberParticles, cudaStream_t st);

    template<class PosIterator, class QuantityIterator>
    void gather(const PosIterator &pos, const QuantityIterator &v,
		real3 *gridData,
		Grid grid, int numberParticles, cudaStream_t st);

    shared_ptr<Kernel> getKernel(){ return this->kernel;}
    
  };

}

#include"IBM.cu"

#endif