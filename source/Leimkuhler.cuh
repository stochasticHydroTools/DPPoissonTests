/*Raul P. Pelaez 2021. A modification of the Leimkuhler Integrator in UAMMD.
the LeimkuhlerMobility Integrator allows to set a self mobility dependent on the height, taking into acocunt the thermal drift that arises in this case.

The specific self mobility is read and interpolated from a file.

This file must have two columns with a list of normalized heights (so Z must go from -1 to 1) and normalized mobilities (i.e. 6*pi*eta*a*M0) in X, Y and Z. The values for each particle will be linearly interpolated from the data provided in the file. The order of the values does not matter. Example:
--- mobility.dat---
-1.0 1.0 1.0 1.0
 0.0 1.0 1.0 1.0
 1.0 1.0 1.0 1.0
-------------------
The above example is equivalent to calling uammd's Leimkuhler integrator.

Small modifications to the SelfMobility class can be made (in particular to its () operator) to make the self mobility vary also in X and/or Y.


*/
#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"misc/TabulatedFunction.cuh"
#include"external/spline.h"
#include <cmath>
using namespace uammd;

//The () operator of this struct must return the normalized self mobility and its derivative when given a position,
// i.e returning 6*pi*eta*a*{M0(x,y,z), \nabla M0(x,y,z)}
class SelfMobility{

  //Reads the height vs the three self mobilities from the file
  auto readMobilityFile(std::string fileName){
    std::ifstream in(fileName);
    std::istream_iterator<real4> begin(in), end;
    std::vector<real4> data{begin, end};
    //The spline library needs input data in ascending order
    std::sort(data.begin(), data.end(),[](real4 a, real4 b){return a.x<b.x;});
    std::vector<double> Z(data.size());
    auto Mx = Z;
    auto My = Mx;
    auto Mz = My;
    std::transform(data.begin(), data.end(), Z.begin(), [](real4 a){return a.x;});
    std::transform(data.begin(), data.end(), Mx.begin(), [](real4 a){return a.y;});
    std::transform(data.begin(), data.end(), My.begin(), [](real4 a){return a.z;});
    std::transform(data.begin(), data.end(), Mz.begin(), [](real4 a){return a.w;});
    return std::make_tuple(Z,Mx,My,Mz);
  }

  //Returns a functor that returns the derivative of "y" at any point via spline interpolation
  template<class Functor> 
  auto computeDerivative(Functor y, int N){
    std::vector<double> derivative(N,0);
    auto x = derivative;
    double h = 2.0/(N-1);
    x[0] = -1;
    derivative[0] = (y(-1+h)-y(-1))/h;
    for(int i = 1; i<N-1; i++){
      real z = -1+h*i;
      x[i] = z;
      derivative[i] = (y(z+h)-y(z-h))/(2.0*h);
    }
    x[N-1] = 1;
    derivative[N-1] = (y(1)-y(1-h))/h;
    tk::spline sy;
    sy.set_points(x, derivative);
    return sy;
  }

  TabulatedFunction<real4> mobilityAndDerivative;
  real Lz;
public:

  SelfMobility(std::string fileName, real Lz):Lz(Lz){
    System::log<System::MESSAGE>("[SelfMobility] Initialized, Lz=%g", Lz);
    auto data = this->readMobilityFile(fileName);
    tk::spline mobilityx, mobilityy, mobilityz;
    mobilityx.set_points(std::get<0>(data), std::get<1>(data));
    mobilityy.set_points(std::get<0>(data), std::get<2>(data));
    mobilityz.set_points(std::get<0>(data), std::get<3>(data));
    constexpr int ntablePoints = 8192;
    tk::spline derivative = this->computeDerivative(mobilityz, 2*std::get<0>(data).size());
    auto allM =[&](real z){return make_real4(mobilityx(z),mobilityy(z),mobilityz(z), derivative(z));};

    std::ofstream out("mob.test");
    int nsam = 1000;
    fori(0, nsam){
      real z = -1 + i*2.0/nsam;
      auto mob = allM(z);
      out<<z<<" "<<mob<<std::endl;
    }
    mobilityAndDerivative = TabulatedFunction<real4>(ntablePoints, -1, 1,
						     allM
			    );
  }

  //The position received by this function will be folded to the main cell.
  __device__ auto operator()(real3 pos){
    real z = (pos.z - floor(pos.z/Lz + real(0.5))*Lz)/Lz; //Height in the [-0.5,0.5] interval
    const auto MandDiffM = mobilityAndDerivative(real(2.0)*z);
    real3 M = make_real3(MandDiffM);
    real diffMz = MandDiffM.w;
    return thrust::make_pair(M, make_real3(0, 0, real(2.0)*diffMz/Lz));
  }

};

//Implements the algorithm in eq. 45 of [2]. Allows for a position-dependent mobility
class LeimkuhlerWithMobility: public BD::BaseBrownianIntegrator{
  std::shared_ptr<SelfMobility> selfMobilityFactor;
public:
  
  LeimkuhlerWithMobility(shared_ptr<ParticleData> pd,
			 shared_ptr<ParticleGroup> pg,
			 shared_ptr<System> sys,
			 Parameters par,
			 std::shared_ptr<SelfMobility> selfMobilityFactor):
    BaseBrownianIntegrator(pd, pg, sys, par),
    selfMobilityFactor(selfMobilityFactor){
    this->seed = sys->rng().next32();
    sys->log<System::MESSAGE>("[BD::LeimkuhlerMobility] Initialized with seed %u", this->seed);
  }

  LeimkuhlerWithMobility(shared_ptr<ParticleData> pd,
			 shared_ptr<System> sys,
			 Parameters par,
			 std::shared_ptr<SelfMobility> selfMobilityFactor):
    LeimkuhlerWithMobility(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par, selfMobilityFactor){}

  void forwardTime() override;

private:
  void updatePositions();
};


namespace Leimkuhler_ns{
  __device__ real3 genNoise(int i, uint stepNum, uint seed){
    Saru rng(i, stepNum, seed);
    return make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
  }
  //This integration scheme allows for a self mobility depending on the position.
  //With the associated non-zero thermal drift term.
  __global__ void integrateGPU(real4* pos,
			       ParticleGroup::IndexIterator indexIterator,
			       const int* originalIndex,
			       const real4* force,
			       real3 Kx, real3 Ky, real3 Kz,
			       real selfMobility,
			       SelfMobility selfMobilityFactor,
			       real* radius,
			       real dt,
			       bool is2D,
			       real temperature,
			       int N,
			       uint stepNum, uint seed){
    uint id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=N) return;
    int i = indexIterator[id];
    real3 R = make_real3(pos[i]);
    const real3 F = make_real3(force[i]);
    const real3 KR = make_real3(dot(Kx, R), dot(Ky, R), dot(Kz, R));
    const auto factor = selfMobilityFactor(R);
    const real m0 = selfMobility*(radius?(real(1.0)/radius[i]):real(1.0));
    const real3 M = m0*factor.first;
    R += dt*( KR + M*F );
    if(temperature > 0){
      int ori = originalIndex[i];
      const auto B = sqrt(real(2.0)*temperature*M*dt);
      const auto dW = genNoise(ori, stepNum, seed);
      // const auto B = sqrt(real(0.5)*temperature*M/dt);
      // const auto dW = genNoise(ori, stepNum, seed) + genNoise(ori, stepNum-1, seed);
      R += B*dW + temperature*dt*m0*factor.second;
    }
    pos[i].x = R.x;
    pos[i].y = R.y;
    if(!is2D)
      pos[i].z = R.z;
  }

}

void LeimkuhlerWithMobility::forwardTime(){
  steps++;
  sys->log<System::DEBUG1>("[BD::Leimkuhler] Performing integration step %d", steps);
  updateInteractors();
  computeCurrentForces();
  updatePositions();
}

void LeimkuhlerWithMobility::updatePositions(){
  int numberParticles = pg->getNumberParticles();
  int BLOCKSIZE = 128;
  uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
  uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
  real * d_radius = getParticleRadiusIfAvailable();
  auto groupIterator = pg->getIndexIterator(access::location::gpu);
  auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
  auto force = pd->getForce(access::location::gpu, access::mode::read);
  auto originalIndex = pd->getIdOrderedIndices(access::location::gpu);
  Leimkuhler_ns::integrateGPU<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
							    groupIterator,
							    originalIndex,
							    force.raw(),
							    Kx, Ky, Kz,
							    selfMobility,
							    *selfMobilityFactor,
							    d_radius,
							    dt,
							    is2D,
							    temperature,
							    numberParticles,
							    steps, seed);
}
