
/*Raul P. Pelaez 2020-2021. DP Poisson test

This file encodes the following simulation:

A group of Gaussian sources of charge with width gw evolve via Brownian Dynamics inside a doubly periodic domain of dimensions Lxy, Lxy, H (periodic in XY and open in Z). 
Optionally, a flag can be passed in the input to enable triply periodic electrostatics.

Sources interact via an electrostatic potential and repell each other via a repulsive LJ-like potential.

There are two (potentially) charged walls on z=-H/2 and z=H/2, this walls have the same charge equal to half of the sum of all charges in the system but with opposite sign (so the overall system is always electroneutral).
Charges are repelled by the wall via the same repulsive potential they have between each other.
The three domains demarcated by the walls (above, between and below) may have different, arbitrary permittivities.

The repulsive potential can be found in RepulsivePotential.cuh (should be available along this file) and has the following form:
U_{LJ}(r) = 4*U0* ( (sigma/r)^{2p} - (sigma/r)^p ) + U0
The repulsive force is then defined by parts using F_{LJ}=-\partial U_{LJ}/\partial r 
F(r<=rm) = F_{LJ}(r=rm);
F(rm<r<=sigma*2^1/p) = F_{LJ}(r);
F(r>sigma*2^1/p) = 0;

USAGE:
This code expects to find a file called data.main in the folder where it is executed.
data.main contains a series of parameters that allow to customize the simulation and must have the following  format:

----data.main starts
#Lines starting with # are ignored
option [argument] #Anything after the argument is ignored
flag #An option that does not need arguments
----end of data.main

The following options are available:

 numberParticles: The number of charges in the simulation
 gw: The Gaussian width of the charges
 
 H: The width of the domain
 Lxy: The dimensions of the box in XY
 permitivity: Permittivity inside the slab (only one used in triply periodic mode)
 permitivityBottom Below z=-H/2. If the value is negative it means metallic boundary (infinite permittivity).  
 permitivityTop Above z=H/2. If the value is negative it means metallic boundary (infinite permittivity).  
 bottomWallSurfaceValue The zero mode value of the Fourier transform of the bottom wall surface value (potential when the boundary is metallic and surface charge otherwise).  

 temperature: Temperature for the Brownian Dynamics integrator, the diffusion coefficient will be D=T/(6*pi*viscosity*hydrodynamicRadius). This temperature is therefore given in units of energy.  
 viscosity: For BD
 hydrodynamicRadius: For BD
 dt: Time step for the BD integrator
 
 U0, sigma, r_m, p: Parameters for the repulsive interaction. If U0=0 the steric repulsion is turned off. 

 wall_U0, wall_sigma, wall_r_m, wall_p Parameters for the ion-wall repulsive interaction.   
 imageDistanceMultiplier Multiplies the distance of the particles to the wall by this amount. For instance, if 2, particles interact with their images, if 1, particles are repelled to the wall (as if the image was at the wall's height)
 noWall Optional, if this flag is present particles will not be repelled by the wall.   

 numberSteps: The simulation will run for this many steps
 printSteps: If greater than 0, the positions and forces will be printed every printSteps steps
 relaxSteps: The simulation will run without printing for this many steps.
 
 outfile: Positions and charge will be written to this file, each snapshot is separated by a #, each line will contain X Y Z Charge. Can be /dev/stdout to print to screen.
 forcefile: Optional, if present forces acting on particles will written to this file.
 fieldfile: Optional, if present electric field acting on particles will written to this file.
 readFile: Optional, if present charge positions will be read from this file with the format X Y Z Charge. numberParticles lines will be read. Can be /dev/stdin to read from pipe. 

 triplyPeriodic: Optional, if this flag is present electrostatics will be solved with a triply periodic spectral ewald solver. Notice that many parameters are not needed in this mode and will be ignored.

 split: The Ewald splitting parameter. It is mandatory if triply periodic mode is enabled.
 Nxy: The number of cells in XY. If this option is present split must NOT be present, it will be computed from this. Nxy can be provided instead of split for doubly periodic mode.

 useMobilityFromFile: Optional, if this option is present, the mobility will depend on the height of the particle according to the data in this file.This file must have two columns with a list of normalized heights (so Z must go from -1 to 1) and normalized mobilities (i.e. 6*pi*eta*a*M0) in X, Y and Z. The values for each particle will be linearly interpolated from the data provided in the file. The order of the values does not matter. Example:
--- mobility.dat---
-1.0 1.0 1.0 1.0
 0.0 1.0 1.0 1.0
 1.0 1.0 1.0 1.0
-------------------

 BrownianUpdateRule: Optional. Can either be EulerMaruyama (default) or Leimkuhler.

 idealParticles: Optional. If this flag is present particles will not interact between them in any way.

 The following accuracy options are optional, the defaults provide a tolerance of 5e-4:
 support: Number of support cells for the interpolation kernel. Default is 10.
 numberStandardDeviations: Gaussian truncation. Default is 4
 tolerance: In doubly periodic, determines the cut off for the near field section of the algortihm. In triply periodic mode this represents the overall accuracy of the solver. Default is 1e-4. 
 upsampling: The relation between the grid cell size and gt=sqrt(gw^2+1/(4*split^2)). h_xy= gt/upsampling. default is 1.2


 
*/
#include"uammd.cuh"
#include"RepulsivePotential.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/SpectralEwaldPoisson.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"BDWithThermalDrift.cuh"
#include"utils/InputFile.h"
#include"Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include <fstream>
#include <limits>
#include<random>
using namespace uammd;
using std::make_shared;
using std::endl;

class RepulsiveWall{
  RepulsivePotentialFunctor::PairParameters params;
  real H;
  real imageDistanceMultiplier = 2.0; //Controls the distance of the image ---->   0  |  0, if this parameter is 2, particles interact with images, if 1, image particles are located at the wall height.
public:
  RepulsiveWall(real H, RepulsivePotentialFunctor::PairParameters ip, real imageDistanceMultiplier):
    H(H),params(ip),imageDistanceMultiplier(imageDistanceMultiplier){}

  __device__ ForceEnergyVirial sum(Interactor::Computables comp, real4 pos /*, real mass */){
    real distanceToImage = abs(abs(pos.z) - H * real(0.5))*imageDistanceMultiplier;
    real fz = RepulsivePotentialFunctor::force(distanceToImage * distanceToImage, params) * distanceToImage;
    ForceEnergyVirial fev;
    fev.force = make_real3(0, 0, fz*(pos.z<0?real(-1.0):real(1.0)));    
    return fev;
  }

  auto getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::gpu, access::read);
    return std::make_tuple(pos.raw());
  }
};
  
struct Parameters{
  int numberParticles;
  real Lxy, H;
  int Nxy = -1;
  int support = 10;
  real numberStandardDeviations = 4;
  real upsampling = 1.2;
  real tolerance = 1e-4;
  real temperature;
  real permitivity, permitivityBottom, permitivityTop;

  real bottomWallSurfaceValue = 0;
  
  int numberSteps, printSteps, relaxSteps;
  real dt, viscosity, hydrodynamicRadius;

  real gw;
  real split = -1;
  real U0, sigma, r_m, p, cutOff;
  real wall_U0, wall_sigma, wall_r_m, wall_p, wall_cutOff;
  real imageDistanceMultiplier;
  
  std::string outfile, readFile, forcefile, fieldfile;

  bool noWall = false;
  bool triplyPeriodic=false;

  std::string mobilityFile, brownianUpdateRule = "EulerMaruyama";
  bool idealParticles=false;
};

struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<thrust::device_vector<real4>> savedPositions;
  Parameters par;
};

Parameters readParameters(std::string fileName);

void initializeParticles(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::write);
  auto charge = sim.pd->getCharge(access::location::cpu, access::mode::write);
  if(sim.par.readFile.empty()){
    std::generate(pos.begin(), pos.end(),
		  [&](){
		    real Lxy = sim.par.Lxy;
		    real H = sim.par.H;
		    real3 p;
		    real pdf;
		    do{
		      p = make_real3(sim.pd->getSystem()->rng().uniform3(-0.5, 0.5))*make_real3(Lxy, Lxy, H-2*sim.par.gw);
		      pdf = 1.0;
		    }while(sim.pd->getSystem()->rng().uniform(0, 1) > pdf);
		    return make_real4(p, 0);
		  });
    fori(0, sim.par.numberParticles){
      charge[i] = ((i%2)-0.5)*2;
    }
  }
  else{
    std::ifstream in(sim.par.readFile);
    fori(0, sim.par.numberParticles){
      in>>pos[i].x>>pos[i].y>>pos[i].z>>charge[i];
      pos[i].w = 0; 
    }
  }
  thrust::copy(pos.begin(), pos.end(), sim.savedPositions->begin());
}

UAMMD initialize(int argc, char *argv[]){
  UAMMD sim;
  auto sys = std::make_shared<System>(argc, argv);
  std::random_device r;
  auto now = static_cast<long long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  sys->rng().setSeed(now);
  std::string datamain = argc>1?argv[1]:"data.main";
  sim.par = readParameters(datamain); 
  sim.pd = std::make_shared<ParticleData>(sim.par.numberParticles, sys);
  sim.savedPositions = std::make_shared<thrust::device_vector<real4>>();
  sim.savedPositions->resize(sim.par.numberParticles);
  initializeParticles(sim); 
  return sim;
}

auto createIntegrator(UAMMD sim){
  typename BD::Parameters par;
  par.temperature = sim.par.temperature;
  par.viscosity = sim.par.viscosity;
  par.hydrodynamicRadius = sim.par.hydrodynamicRadius;
  par.dt = sim.par.dt;
  std::shared_ptr<Integrator> integrator;
  if(sim.par.mobilityFile.empty()){
    if(sim.par.brownianUpdateRule== "Leimkuhler"){
      using BDMethod = BD::Leimkuhler;
      integrator =  std::make_shared<BDMethod>(sim.pd, par);
    }
    else if(sim.par.brownianUpdateRule== "EulerMaruyama"){
      using BDMethod = BD::EulerMaruyama;
      integrator =  std::make_shared<BDMethod>(sim.pd, par);
    }
  }
  else{
    auto selfMobilityFactor = std::make_shared<SelfMobility>(sim.par.mobilityFile, sim.par.H);
    integrator = std::make_shared<BDWithThermalDrift>(sim.pd, par,
						      sim.par.brownianUpdateRule,
						      selfMobilityFactor);
  }
  return integrator;
}

auto createDoublyPeriodicElectrostaticInteractor(UAMMD sim){  
  DPPoissonSlab::Parameters par;
  par.Lxy = make_real2(sim.par.Lxy);
  par.H = sim.par.H;
  DPPoissonSlab::Permitivity perm;
  perm.inside = sim.par.permitivity;
  perm.top = sim.par.permitivityTop;
  if(sim.par.permitivityTop<0){ //Metallic boundary
    perm.top = std::numeric_limits<real>::infinity();
  }
  perm.bottom = sim.par.permitivityBottom;
  if(sim.par.permitivityBottom<0){ //Metallic boundary
    perm.bottom = std::numeric_limits<real>::infinity();
  }
  par.permitivity = perm;
  par.gw = sim.par.gw;
  par.tolerance = sim.par.tolerance;
  if(sim.par.split>0){
    par.split = sim.par.split;
  }
  if(sim.par.upsampling > 0){
    par.upsampling=sim.par.upsampling;
  }
  if(sim.par.numberStandardDeviations > 0){
    par.numberStandardDeviations=sim.par.numberStandardDeviations;
  }
  if(sim.par.support > 0){
    par.support=sim.par.support;
  }
  if(sim.par.Nxy > 0){
    if(sim.par.split>0){
      System::log<System::CRITICAL>("ERROR: Cannot set both Nxy and split at the same time");
    }
    par.Nxy = sim.par.Nxy;
  }
  par.support = sim.par.support;
  par.numberStandardDeviations = sim.par.numberStandardDeviations;
  auto dppoisson = std::make_shared<DPPoissonSlab>(sim.pd, par);
  if(sim.par.bottomWallSurfaceValue){
    System::log<System::MESSAGE>("[DPPoisson] Setting the bottom wall zero mode Fourier value to %g", sim.par.bottomWallSurfaceValue);
    dppoisson->setSurfaceValuesZeroModeFourier({0, 0, sim.par.bottomWallSurfaceValue, 0});
  }
  return dppoisson;
}

auto createTriplyPeriodicElectrostaticInteractor(UAMMD sim){
  Poisson::Parameters par;
  par.box = Box(make_real3(sim.par.Lxy, sim.par.Lxy, sim.par.H));
  par.epsilon = sim.par.permitivity;
  par.gw = sim.par.gw;
  par.tolerance = sim.par.tolerance;
  if(sim.par.split<0){
    System::log<System::CRITICAL>("ERROR: Triply periodic mode needs an splitting parameter (Nxy wont do)");
  }
  par.split = sim.par.split;
  return std::make_shared<Poisson>(sim.pd, par);
}

auto createWallRepulsionInteractor(UAMMD sim){
  RepulsivePotentialFunctor::PairParameters potpar;
  potpar.cutOff2 = sim.par.wall_cutOff*sim.par.wall_cutOff;
  potpar.sigma = sim.par.wall_sigma;
  potpar.U0 = sim.par.wall_U0;
  potpar.r_m = sim.par.wall_r_m;
  potpar.p = sim.par.wall_p;
  return make_shared<ExternalForces<RepulsiveWall>>(sim.pd, make_shared<RepulsiveWall>(sim.par.H, potpar, sim.par.imageDistanceMultiplier));
}

auto createPotential(UAMMD sim){
  auto pot = std::make_shared<RepulsivePotential>();
  RepulsivePotential::InputPairParameters ppar;
  ppar.cutOff = sim.par.cutOff;
  ppar.U0 = sim.par.U0;
  ppar.sigma = sim.par.sigma;
  ppar.r_m = sim.par.r_m;
  ppar.p = sim.par.p;
  System::log<System::MESSAGE>("Repulsive rcut: %g", ppar.cutOff);
  pot->setPotParameters(0, 0, ppar);
 return pot;
}

template<class UsePotential> auto createShortRangeInteractor(UAMMD sim){
  auto pot = createPotential(sim);
  using SR = PairForces<UsePotential>;
  typename SR::Parameters params;
  real Lxy = sim.par.Lxy;
  real H = sim.par.H;
  params.box = Box(make_real3(Lxy, Lxy, H));
  if(not sim.par.triplyPeriodic){
    params.box.setPeriodicity(1,1,0);
  }
  auto pairForces = std::make_shared<SR>(sim.pd, params, pot);
  return pairForces;
}

void writeSimulation(UAMMD sim, std::vector<real4> fieldAtParticles){ 
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::read);
  auto charge = sim.pd->getCharge(access::location::cpu, access::mode::read);
  auto force = sim.pd->getForce(access::location::cpu, access::mode::read);
  static std::ofstream out(sim.par.outfile);
  static std::ofstream outf(sim.par.forcefile);
  static std::ofstream outfield(sim.par.fieldfile);
  Box box(make_real3(sim.par.Lxy, sim.par.Lxy, sim.par.H));
  box.setPeriodicity(1,1,sim.par.triplyPeriodic?1:0);
  real3 L = box.boxSize;
  out<<"#Lx="<<L.x*0.5<<";Ly="<<L.y*0.5<<";Lz="<<L.z*0.5<<";"<<std::endl;
  if(outf.good())outf<<"#"<<std::endl;
  if(outfield.good())outfield<<"#"<<std::endl;
  fori(0, sim.par.numberParticles){
    real3 p = box.apply_pbc(make_real3(pos[i]));
    real q = charge[i];
    out<<std::setprecision(2*sizeof(real))<<p<<" "<<q<<"\n";
    if(outf.good()){
      outf<<std::setprecision(2*sizeof(real))<<force[i]<<"\n";
    }
    if(outfield.good() and fieldAtParticles.size()>0){
      outfield<<std::setprecision(2*sizeof(real))<<fieldAtParticles[i]<<"\n";
    }
  }
  out<<std::flush;
}

struct CheckOverlap {
  real H;
  CheckOverlap(real H):H(H){
    
  }

  __device__ bool operator()(real4 p){
    return abs(p.z) >= (real(0.5)*H);
  }  
  
};

bool checkWallOverlap(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::read);
  //int overlappingCharges = thrust::count_if(thrust::cuda::par, pos.begin(), pos.end(), CheckOverlap(sim.par.H));
  //return overlappingCharges > 0;
  auto overlappingPos = thrust::find_if(thrust::cuda::par, pos.begin(), pos.end(), CheckOverlap(sim.par.H));
  return overlappingPos != pos.end();
}

void restoreLastSavedConfiguration(UAMMD sim) {
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::write);  
  thrust::copy(thrust::cuda::par, sim.savedPositions->begin(), sim.savedPositions->end(), pos.begin());  
}

void saveConfiguration(UAMMD sim) {
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::read);
  thrust::copy(thrust::cuda::par, pos.begin(), pos.end(), sim.savedPositions->begin()); 
}

int main(int argc, char *argv[]){  
  auto sim = initialize(argc, argv);
  auto bd = createIntegrator(sim);
  std::shared_ptr<DPPoissonSlab> dpslab;
  if(not sim.par.idealParticles){
    if(sim.par.triplyPeriodic){
      bd->addInteractor(createTriplyPeriodicElectrostaticInteractor(sim));
    }
    else{
      dpslab = createDoublyPeriodicElectrostaticInteractor(sim);
      bd->addInteractor(dpslab);
    }
    if(sim.par.U0 > 0){
      bd->addInteractor(createShortRangeInteractor<RepulsivePotential>(sim));
    }
  }
  if(not sim.par.noWall){
    bd->addInteractor(createWallRepulsionInteractor(sim));
  }
  int numberRetries=0;
  int numberRetriesThisStep=0;
  int lastStepSaved=0;
  constexpr int saveRate = 100;
  constexpr int maximumRetries = 1e6;
  constexpr int maximumRetriesPerStep=1e4;
  forj(0, sim.par.relaxSteps){
    bd->forwardTime();
    if(not sim.par.triplyPeriodic){
      if(checkWallOverlap(sim)){
	numberRetries++;
	if(numberRetries>maximumRetries){
	  throw std::runtime_error("Too many steps with wall overlapping charges detected, aborting run");
	}
	numberRetriesThisStep++;
	if(numberRetriesThisStep>maximumRetriesPerStep){
	  throw std::runtime_error("Cannot recover from configuration with wall overlapping charges, aborting run");
	}
	j=lastStepSaved;
	restoreLastSavedConfiguration(sim);
	continue;
      }
      if(j%saveRate==0){
	numberRetriesThisStep = 0;
	lastStepSaved=j;
	saveConfiguration(sim);
      }
    }
  }
  Timer tim;
  tim.tic();
  lastStepSaved=0;
  forj(0, sim.par.numberSteps){
    bd->forwardTime();
    if(not sim.par.triplyPeriodic){
      if(checkWallOverlap(sim)){
	numberRetries++;
	if(numberRetries>maximumRetries){
	  throw std::runtime_error("Too many steps with wall overlapping charges detected, aborting run");
	}
	numberRetriesThisStep++;
	if(numberRetriesThisStep>maximumRetriesPerStep){
	  throw std::runtime_error("Cannot recover from configuration with wall overlapping charges, aborting run");
	}
	j=lastStepSaved;
	restoreLastSavedConfiguration(sim);
	continue;
      }
      if(j%saveRate==0){
	numberRetriesThisStep=0;
	lastStepSaved=j;
	saveConfiguration(sim);
      }
    }
    if(sim.par.printSteps > 0 and j%sim.par.printSteps==0){
      std::vector<real4> fieldAtParticles;
      if(not sim.par.fieldfile.empty() and dpslab){
	auto d_field = dpslab->computeFieldAtParticles();
	fieldAtParticles.resize(d_field.size());
	thrust::copy(d_field.begin(), d_field.end(), fieldAtParticles.begin());
      }
      writeSimulation(sim, fieldAtParticles);
      numberRetriesThisStep=0;
      lastStepSaved=j;
      saveConfiguration(sim);
    }
  }
  System::log<System::MESSAGE>("Number of rejected configurations: %d (%g%% of total)", numberRetries, (double)numberRetries/(sim.par.numberSteps + sim.par.relaxSteps)*100.0);
  auto totalTime = tim.toc();
  System::log<System::MESSAGE>("mean FPS: %.2f", sim.par.numberSteps/totalTime);
  return 0;
}

Parameters readParameters(std::string datamain){
  InputFile in(datamain);
  Parameters par;
  if(in.getOption("triplyPeriodic", InputFile::Optional)){
    par.triplyPeriodic= true;
  }

  in.getOption("Lxy", InputFile::Required)>>par.Lxy;
  in.getOption("H", InputFile::Required)>>par.H;
  in.getOption("numberSteps", InputFile::Required)>>par.numberSteps;
  in.getOption("printSteps", InputFile::Required)>>par.printSteps;
  in.getOption("relaxSteps", InputFile::Required)>>par.relaxSteps;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("numberParticles", InputFile::Required)>>par.numberParticles;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>par.hydrodynamicRadius;
  in.getOption("outfile", InputFile::Required)>>par.outfile;
  in.getOption("forcefile", InputFile::Optional)>>par.forcefile;
  in.getOption("fieldfile", InputFile::Optional)>>par.fieldfile;
  in.getOption("U0", InputFile::Required)>>par.U0;
  in.getOption("r_m", InputFile::Required)>>par.r_m;
  in.getOption("p", InputFile::Required)>>par.p;
  in.getOption("sigma", InputFile::Required)>>par.sigma;
  in.getOption("readFile", InputFile::Optional)>>par.readFile;
  in.getOption("gw", InputFile::Required)>>par.gw;
  in.getOption("tolerance", InputFile::Optional)>>par.tolerance;
  in.getOption("permitivity", InputFile::Required)>>par.permitivity;
  if(not par.triplyPeriodic){
    in.getOption("permitivityTop", InputFile::Required)>>par.permitivityTop;
    in.getOption("permitivityBottom", InputFile::Required)>>par.permitivityBottom;
  }
  if(not par.triplyPeriodic){
    in.getOption("split", InputFile::Optional)>>par.split;
  }
  else{
    in.getOption("split", InputFile::Required)>>par.split;
  }
  in.getOption("Nxy", InputFile::Optional)>>par.Nxy;
  in.getOption("support", InputFile::Optional)>>par.support;
  in.getOption("numberStandardDeviations", InputFile::Optional)>>par.numberStandardDeviations;
  in.getOption("upsampling", InputFile::Optional)>>par.upsampling;
  if(in.getOption("noWall", InputFile::Optional)){
    par.noWall= true;
  }
  else{
    in.getOption("wall_U0", InputFile::Required)>>par.wall_U0;
    in.getOption("wall_r_m", InputFile::Required)>>par.wall_r_m;
    in.getOption("wall_p", InputFile::Required)>>par.wall_p;
    in.getOption("wall_sigma", InputFile::Required)>>par.wall_sigma;
    in.getOption("imageDistanceMultiplier", InputFile::Required)>>par.imageDistanceMultiplier;
    par.wall_cutOff = par.wall_sigma*pow(2,1.0/par.wall_p);
  }  
  if(par.split < 0 and par.Nxy < 0){
    System::log<System::CRITICAL>("ERROR: I need either Nxy or split");    
  }
  par.cutOff = par.sigma*pow(2,1.0/par.p);
  in.getOption("useMobilityFromFile", InputFile::Optional)>>par.mobilityFile;
  in.getOption("BrownianUpdateRule", InputFile::Optional)>>par.brownianUpdateRule;
  if(in.getOption("idealParticles", InputFile::Optional))
    par.idealParticles = true;
  in.getOption("bottomWallSurfaceValue", InputFile::Optional)>>par.bottomWallSurfaceValue;   
  return par;
}



