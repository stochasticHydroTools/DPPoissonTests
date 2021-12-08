#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"misc/TabulatedFunction.cuh"
#include"external/spline.h"
using namespace uammd;

//The () operator of this struct must return the normalized self mobility and its derivative when given a position,
// i.e returning 6*pi*eta*a*{M0(x,y,z), \nabla M0(x,y,z)}
class SelfMobility{

  auto readMobilityFile(std::string fileName){
    std::ifstream in(fileName);
    std::istream_iterator<real2> begin(in), end;
    std::vector<real2> data{begin, end};
    std::sort(data.begin(), data.end(),[](real2 a, real2 b){return a.x<b.x;});
    std::vector<double> Z(data.size());
    auto M = Z;
    std::transform(data.begin(), data.end(), Z.begin(), [](real2 a){return a.x;});
    std::transform(data.begin(), data.end(), M.begin(), [](real2 a){return a.y;});
    return std::make_pair(Z,M);
  }

  template<class Functor> 
  auto computeDerivative(Functor &y, int N){
    std::vector<double> derivative(N,0);
    auto x = derivative;
    double h = 2.0/(N-1);
    x[0] = -1;
    derivative[0] = (y(-1+h)-y(-1))/h;
    for(int i = 1; i<N-1; i++){
      real z = -1+h;
      x[i] = z;
      derivative[i] = (y(z+h)-y(z-h))/h;
    }
    x[N-1] = 1;
    derivative[N-1] = (y(1)-y(1-h))/h;
    tk::spline sy;
    sy.set_points(x, derivative);
    return sy;
  }

  TabulatedFunction<real2> mobilityAndDerivative;
  real Lz;
public:

  SelfMobility(std::string fileName, real Lz):Lz(Lz){
    auto data = this->readMobilityFile(fileName);
    tk::spline mobility;
    mobility.set_points(data.first, data.second);
    constexpr int ntablePoints = 8192;
    tk::spline derivative = this->computeDerivative(mobility, ntablePoints);
    mobilityAndDerivative = TabulatedFunction<real2>(ntablePoints, -1, 1,
			    [&](real z){
			      return make_real2(mobility(z), derivative(z));
			    });
  }
  //The position received by this function will NOT be folded to the main cell.
  __device__ real4 operator()(real3 pos){
    const real z = 2.0*pos.z/Lz; //Height in the [-1,1] interval
    const real2 MandDiffM = mobilityAndDerivative(z);
    const real M = MandDiffM.x;
    const real diffMz = MandDiffM.y;
    return {M,0,0,diffMz};
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
    sys->log<System::MESSAGE>("[BD::Leimkuhler] Initialized");
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
    const real M = selfMobility*(radius?(real(1.0)/radius[i]):real(1.0))*factor.x;
    R += dt*( KR + M*F );
    if(temperature > 0){
      int ori = originalIndex[i];
      const real B = sqrt(real(0.5)*temperature*M*dt);
      const real3 dW = genNoise(ori, stepNum, seed) + genNoise(ori, stepNum-1, seed);
      R += B*dW;
      R += temperature*make_real3(factor.y, factor.z, factor.w);
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
