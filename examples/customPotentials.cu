/*Raul P. Pelaez 2021.
Custom Potential examples showing the different computations the Potential interface can expose. Several Potentials of increasing complexity are available. 
A data.main.potentials file can be used to provide some parameters (it will be autogenerated if it is not present).

Potentials provide Transversers to PairForces in order to compute forces, energies and/or both at the same time. Many aspects of the Potential and Transverser interfaces are optional and provide default behavior, when a function is optional it will be denoted as such in this header.

The Potential interface requires a given class/struct to provide the following public member functions:
Required members:
 real getCutOff(); //The maximum cut-off the potential requires.
 ForceTransverser getForceTransverser(Box box, shared_ptr<ParticleData> pd); //Provides a Transverser that computes the force
Optional members:
 EnergyTransverser getEnergyTransverser(Box box, shared_ptr<ParticleData> pd); //Provides a Transverser that computes the energy
 //If not present defaults to every interaction having zero energy contribution
 ForceEnergyTransverser getForceEnergyTransverser(Box box, shared_ptr<ParticleData> pd); //Provides a Transverser that computes the force and energy at the same time
 //If not present defaults to sequentially computing force and energy one after the other.
A struct/class adhering to the Potential interface can also be ParameterUpdatable[1].

The type(s) returned by these functions must adhere to the Transverser interface described below.
For each particle to be processed the Transverser will be called for:
  -Setting the initial value of the interaction result (function zero)
  -Fetching the necesary data to process a pair of particles (function getInfo)
  -Compute the interaction between the particle and each of its neighbours (function compute)
  -Accumulate/reduce the result for each neighbour  (function accumulate)
  -Set/write/handle the accumulated result for all neighbours (function set)
The same Transverser instance will be used to process every particle in an arbitrary order. Therefore, the Transverser must not assume it is bound to a specific particle.
   
The Transverser interface requires a given class/struct to provide the following public device (unless prepare that must be a host function) member functions:

    Compute zero(); 
      -This function returns the initial value of the computation, for example {0,0,0} when computing the force. 
      -The returning type, Compute, must be a POD type (just an aggregate of plain types), for example a real when computing energy. Furthemore it must be the same type returned by the "compute" member.
      -This function is optional and defaults to zero initialization (it will return Compute() which works even for POD types).
    
    Info getInfo(int particle_id);
      -Will be called for each particle to be processed and returns the per-particle data necessary for the interaction with another particle (except the position which is always available). For example the mass in a gravitational interaction or the particle index for some custom interaction.
      -The returning type, Info, must be a POD type (just an aggregate of plain types), for example a real for gravitation.
      -This function is optional and if not present it is assumed the only per-particle data required is the position. 
       -In this case the function "compute" must only have the first two arguments.

    Compute compute(real4 position_i, real4 position_j, Info info_i, Info info_j)
      -For a pair of particles characterized by position and info this function must return the result of the interaction for that pair of particles.
      -The last two arguments must be present only when getInfo is defined.
      -The returning type, Compute, must be a POD type (just an aggregate of plain types), for example a real when computing energy. 

    void accumulate(Compute &total, const Compute &current);
      -This function will be called after "compute" for each neighbour with its result and the accumulated result.
      -It is expected that this function modifies "total" as necessary given the new data in "current".
      -The first time it is called "total" will be have the value as given by the "zero" function.
      -This function is optional and defaults tu summation: total = total + current. Notice that this will fail for non trivial types.

    void set(int particle_index, Compute &total);
     -After calling compute for all neighbours this function will be called with the contents of "total" after the last call to "accumulate".
     -Can be used to, for example, write the final result to main memory.
      
    void prepare(std::shared_ptr<ParticleData> pd);
     -This function will be called one time in the CPU side just before processing the particles.
     -This function is optional and defaults to simply nothing.

[1] https://github.com/RaulPPelaez/UAMMD/wiki/Parameter-Updatable 
 */
#include"uammd.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"utils/InputFile.h"
#include"utils/InitialConditions.cuh"
#include"Interactor/Interactor.cuh"
#include"Integrator/VerletNVT.cuh"
#include<fstream>

using namespace uammd;
using std::endl;
using std::make_shared;
//Parameters that can be read from a file, see readParameters
real3 boxSize;
real dt;
std::string outputFile, chosenPotential;
int numberParticles;
int numberSteps, printSteps;
real temperature, viscosity;

//Some functions to compute forces/energies
__device__ real lj_force(real r2){
  const real invr2 = real(1.0)/r2;
  const real invr6 = invr2*invr2*invr2;
  const real fmoddivr = (real(-48.0)*invr6 + real(24.0))*invr6*invr2;
  return fmoddivr;
}

__device__ real lj_energy(real r2){
  const real invr2 = real(1.0)/r2;
  const real invr6 = invr2*invr2*invr2;
  return real(4.0)*(invr6 - real(1.0))*invr6;
}

__device__ real wca_force(real r2){
  if(r2>= real(1.259921049894873)) //2^(2/6)
    return 0;
  const real invr2 = real(1.0)/r2;
  const real invr6 = invr2*invr2*invr2;
  const real fmoddivr = thrust::max(real(200),(real(-48.0)*invr6 + real(24.0))*invr6*invr2); 
  return fmoddivr;
}

__device__ real gravity_force(real r2){
  const real invr2 = real(1.0)/r2;
  const real fmoddivr = invr2*rsqrt(r2);
  return fmoddivr;
}

/**------------------------------------------------------------------------------------------**/
//A simple LJ Potential, can compute force, energy or both at the same time.
struct SimpleLJ{
  real rc = 2.5;
  //A host function returning the maximum required cut off for the interaction
  real getCutOff(){
    return rc;
  }
  //A Transverser for computing both energy and force, this same Transverser can be used to compute either force, energy or both at the same time. It is the simplest form of Transverser as it only provides the "compute" and "set" functions
  //When constructed, if the i_force or i_energy pointers are null that computation will be avoided.
  //Notice that it is not required that this struct is defined inside the Potential, it is only required that the functions get*Transverser provide it.
  struct ForceEnergy{
    real4 *force;
    real* energy;
    Box box;
    real rc;
    ForceEnergy(Box i_box, real i_rc, real4* i_force, real* i_energy):
      box(i_box),
      rc(i_rc),
      force(i_force),
      energy(i_energy){
      //All members will be available in the device functions
    }
    //For each pair computes and returns the LJ force and/or energy based only on the positions
    __device__ real4 compute(real4 pi, real4 pj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      if(r2>0 and r2< rc*rc){
	return make_real4(force?(lj_force(r2)*rij):real3(),energy?lj_energy(r2):0);
      }
      return real4();
    }
    //There is no "accumulate" function so, for each particle, the result of compute for every neighbour will be summed
    //There is no "zero" function so the total result starts being real4() (or {0,0,0,0}).
    //The "set" function will be called with the accumulation of the result of "compute" for all neighbours. 
    __device__ void set(int id, real4 total){
      //Write the total result to memory if the pointer was provided
      if(force)  force[id] += make_real4(total.x, total.y, total.z, 0);
      if(energy) energy[id] += total.w;
    }
  };

  //Return an instance of the Transverser that will compute only the force (because the energy pointer is null)
  ForceEnergy getForceTransverser(Box box, std::shared_ptr<ParticleData> pd){
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();    
    return ForceEnergy(box, rc, force, nullptr);
  }

  //These two functions can be ommited if one is not interested in the energy as explained in the header.
  //They provide instances of the Transverser that compute either the energy or the force and energy.
  //Notice that it is not required that the return type is the same in all three cases. Different Transversers can be used in each case.
  ForceEnergy getEnergyTransverser(Box box, std::shared_ptr<ParticleData> pd){
    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite).raw();   
    return ForceEnergy(box, rc, nullptr, energy);
  }
  
  ForceEnergy getForceEnergyTransverser(Box box, std::shared_ptr<ParticleData> pd){
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();
    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite).raw();
    return ForceEnergy(box, rc, force, energy);
  }
  
};
/**------------------------------------------------------------------------------------------**/
//Similar to the previous one, but this time the Transversers will count and store the number of neighbours in addition to computing force. For simplicity only the force is computed now.
struct SimpleLJWithNeighbourCounting{
  thrust::device_vector<int> numberNeighboursPerParticle;
  real rc = 2.5;
  //A host function returning the maximum required cut off for the interaction
  real getCutOff(){
    return rc;
  }
  //A Transverser for computing both energy and force, this same Transverser can be used to compute either force, energy or both at the same time. This time we also need to define the "accumulate" function since "compute" returns a non trivial type.
  //When constructed, if the i_force or i_energy pointers are null that computation will be avoided.
  //Notice that it is not required that this struct is defined inside the Potential, it is only required that the functions get*Transverser provide it.
  struct Force{
    real4 *force;
    int *nneigh;
    Box box;
    real rc;
    Force(Box i_box, real i_rc, real4* i_force,int *nneigh):
      box(i_box),
      rc(i_rc),
      force(i_force),
      nneigh(nneigh){
      //All members will be available in the device functions
    }
    struct Compute{
      real3 force;
      int numberNeighbours;
    };
    //There is no "zero" function so the total result starts being Compute() (or {0,0,0,0}).
    
    //For each pair computes and returns the LJ force and/or energy based only on the positions, it also counts a neighbour if the particle is closer than rcut
    __device__ Compute compute(real4 pi, real4 pj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      if(r2>0 and r2< rc*rc){
	return {lj_force(r2)*rij, 1};
      }
      return {real3(), 0};
    }
    __device__ void accumulate(Compute &total, const Compute &current){
      total.force += current.force;
      total.numberNeighbours += current.numberNeighbours;
    }
    //The "set" function will be called with the accumulation of the result of "compute" for all neighbours. 
    __device__ void set(int index, Compute total){
      force[index] += make_real4(total.force);
      nneigh[index] = total.numberNeighbours;
    }
  };

  //Return an instance of the Transverser that will compute only the force, also will print the average number of neighbours based on the previous time it was called.
  Force getForceTransverser(Box box, std::shared_ptr<ParticleData> pd){
    int N = pd->getNumParticles();
    numberNeighboursPerParticle.resize(N,0);
    int avgneigh = int(thrust::reduce(numberNeighboursPerParticle.begin(), numberNeighboursPerParticle.end())/double(N));
    System::log<System::MESSAGE>("Average number of neighbours in the previous step: %d", avgneigh);
    thrust::fill(numberNeighboursPerParticle.begin(), numberNeighboursPerParticle.end(), 0);
    int* nneigh = thrust::raw_pointer_cast(numberNeighboursPerParticle.data());
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();    
    return Force(box, rc, force, nneigh);
  }
  
};
/**------------------------------------------------------------------------------------------**/
//A gravitational Potential, similar to SimpleLJ but needs the particle masses in addition to the positions. For simplicity only the force is provided. It is coupled with a short range repulsion and if used it will probably result in all particles in the system forming a sphere.
class GravityPotential{
public:
  real getCutOff(){
    return boxSize.x;
  }
  struct Force{
    real4 *force;
    real* mass;
    Box box;
    Force(Box i_box):
      box(i_box){
      //All members will be available in the device functions
    }
    //Get the mass of the particles, notice that this could also be donde via the constructor. 
    void prepare(std::shared_ptr<ParticleData> pd){
      this->force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();
      this->mass = pd->getMass(access::location::gpu, access::mode::read).raw();
    }
    
    //Fetch the mass of each particle
    __device__ real getInfo(int particle_index){
      return mass[particle_index];
    }
    
    //For each pair computes and returns the gravitational force.
    //Notice that the existance of getInfo requires two additional arguments to compute
    __device__ real3 compute(real4 pi, real4 pj, real mass_i, real mass_j){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      //A short range wca repulsion is added to avoid overlapping
      if(r2 > 0)
	return (gravity_force(r2)*mass_i*mass_j+wca_force(r2))*rij;
      else
	return real3();
    }
    //There is no "accumulate" function so, for each particle, the result of compute for every neighbour will be summed
    //There is no "zero" function so the total result starts being real3() (or {0,0,0}).
    //The "set" function will be called with the accumulation of the result of "compute" for all neighbours. 
    inline __device__ void set(int index, real3 total){
      //Write the total result to memory
      force[index] += make_real4(total);
    }
  };
  //Return an instance of the Transverser that will compute only the force (because the energy pointer is null)
  Force getForceTransverser(Box box, std::shared_ptr<ParticleData> pd){
    return Force(box);
  }
};
/**------------------------------------------------------------------------------------------**/
//A Potential similar to the previous one, but this time it requires both the mass and id of the particles because particles will only interact gravitantionally if their indices follow a certain rule. Furthermore the potential needs to be aware of the simulation time because the rule changes over time.
//ParameterUpdatable allows to define the function updateSimulationTime, which will be called each time the simulation time changes (AKA every step). This Potential will probably result in a very boring dynamics.
class CustomIntricatePotential: public ParameterUpdatable{
  int currentStep; //current simulation step;
public:
  real getCutOff(){
    return boxSize.x; //boxSize is a global variable
  }

  //Part of the ParameterUpdatable interface, will be called every time the time changes
  void updateSimulationTime(real newTime){
    currentStep = newTime/dt; //dt is a global variable 
  }
  
  struct Force{
    real4 *force;
    real* mass;
    int *index2id;
    Box box;
    int currentStep;
    Force(Box i_box, int cs):
      box(i_box),currentStep(cs){
      //All members will be available in the device functions
    }
    //Get the mass of the particles, notice that this could also be donde via the constructor. 
    void prepare(std::shared_ptr<ParticleData> pd){
      this->force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();
      this->mass = pd->getMass(access::location::gpu, access::mode::read).raw();
      this->index2id = pd->getId(access::location::gpu, access::mode::read).raw();
    }

    struct Info{
      real mass;
      int id;
    };
    //Fetch the mass of each particle
    __device__ Info getInfo(int particle_index){
      return {mass[particle_index], index2id[particle_index]};
    }
    
    //For each pair computes and returns the gravitational force.
    //Notice that the existance of getInfo requires two additional arguments to compute
    __device__ real3 compute(real4 pi, real4 pj, Info infoi, Info infoj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      //A short range wca repulsion is added to avoid overlapping
      real fmod = (r2>0)?wca_force(r2):0;
      //Just a dumb interaction rule that depends on time and particle ids. 
      bool areIdsEven = infoi.id%2== 0 and infoj.id%2==0;
      bool isGravity = ((currentStep/1000)%2==0)?areIdsEven:true;      
      fmod += (r2>0 and isGravity)?(gravity_force(r2)*infoi.mass*infoj.mass):0;
      return rij*fmod;
    }
    //There is no "accumulate" function so, for each particle, the result of compute for every neighbour will be summed
    //There is no "zero" function so the total result starts being real3() (or {0,0,0}).
    //The "set" function will be called with the accumulation of the result of "compute" for all neighbours. 
    inline __device__ void set(int index, real3 total){
      //Write the total result to memory
      force[index] += make_real4(total);
    }
  };
  //Return an instance of the Transverser that will compute only the force (because the energy pointer is null)
  Force getForceTransverser(Box box, std::shared_ptr<ParticleData> pd){
    return Force(box, currentStep);
  }
};
/**------------------------------------------------------------------------------------------**/
//Below is a standard UAMMD simulation creation and execution

void initializeParticles(std::shared_ptr<ParticleData> pd){
  Box box(boxSize);
  auto pos = pd->getPos(access::location::cpu, access::mode::write);
  auto initial =  initLattice(box.boxSize, numberParticles, fcc);
  std::transform(initial.begin(), initial.end(), pos.begin(), [&](real4 p){p.w = 0;return p;});
  auto mass = pd->getMass(access::location::cpu, access::mode::write);
  std::fill(mass.begin(), mass.end(), 1.0);
}

shared_ptr<Integrator> createIntegrator(shared_ptr<ParticleData> pd, shared_ptr<System> sys){
  using NVT = VerletNVT::GronbechJensen;
  NVT::Parameters par;
  par.temperature = temperature;
  par.dt = dt;
  par.viscosity = viscosity;
  return make_shared<NVT>(pd, sys, par);
}

template<class UsePotential>
shared_ptr<Interactor> createPairForcesWithPotential(shared_ptr<ParticleData> pd, shared_ptr<System> sys){
  using PF = PairForces<UsePotential>;
  typename PF::Parameters par;
  par.box = Box(boxSize);
  auto pot = std::make_shared<UsePotential>();
  return std::make_shared<PF>(pd, sys, par, pot);  
}

shared_ptr<Interactor> createInteraction(shared_ptr<ParticleData> pd, shared_ptr<System> sys){
  if(chosenPotential.compare("SimpleLJ") == 0)
    return createPairForcesWithPotential<SimpleLJ>(pd, sys);
  if(chosenPotential.compare("SimpleLJWithNeighbourCounting") == 0)
    return createPairForcesWithPotential<SimpleLJWithNeighbourCounting>(pd, sys);
  if(chosenPotential.compare("GravityPotential") == 0)
    return createPairForcesWithPotential<GravityPotential>(pd, sys);
  if(chosenPotential.compare("CustomIntricatePotential") == 0)
    return createPairForcesWithPotential<CustomIntricatePotential>(pd, sys);
  else
    sys->log<System::CRITICAL>("Invalid Potential selected in data.main.potentials!");
  return nullptr;
}

void runSimulation(shared_ptr<Integrator> integrator, shared_ptr<ParticleData> pd, shared_ptr<System> sys){
  std::ofstream out(outputFile);
  Timer tim;
  tim.tic();
  Box box(boxSize);
  forj(0, numberSteps){
    integrator->forwardTime();
    if(printSteps > 0 and j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);      
      out<<"#Lx="<<0.5*boxSize.x<<";Ly="<<0.5*boxSize.y<<";Lz="<<0.5*boxSize.z<<";"<<endl;
      fori(0, numberParticles){
	real4 pc = pos[sortedIndex[i]];
	real3 p = box.apply_pbc(make_real3(pc));
	out<<p<<" "<<0.5<<" "<<0<<"\n";
      }
      out<<std::flush;
    }
    if(j%500 == 0){
      pd->sortParticles();
    }
  }
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps/totalTime);
}

void readParameters(std::shared_ptr<System> sys, std::string file);

int main(int argc, char *argv[]){
  auto sys = std::make_shared<System>(argc, argv);
  readParameters(sys, "data.main.potentials");
  auto pd = std::make_shared<ParticleData>(numberParticles, sys);
  initializeParticles(pd);
  auto verlet = createIntegrator(pd, sys);
  verlet->addInteractor(createInteraction(pd, sys));
  runSimulation(verlet, pd, sys);
  sys->finish();
  return 0;
}

void generateDefaultParameters(std::string file){
  std::ofstream default_options(file);
  default_options<<"#possible potential options: SimpleLJ SimpleLJWithNeighbourCounting GravityPotential CustomIntricatePotential"<<std::endl;
  default_options<<"potential SimpleLJ"<<std::endl;
  default_options<<"boxSize 32 32 32"<<std::endl;
  default_options<<"numberParticles 16384"<<std::endl;
  default_options<<"outputFile /dev/stdout"<<std::endl;
  default_options<<"dt 0.01"<<std::endl;
  default_options<<"numberSteps 500"<<std::endl;
  default_options<<"printSteps -1"<<std::endl;
  default_options<<"temperature 1.0"<<std::endl;
  default_options<<"viscosity 1"<<std::endl;
}

void readParameters(std::shared_ptr<System> sys, std::string file){
  if(!std::ifstream(file).good()){
    generateDefaultParameters(file);
  }
  InputFile in(file, sys);
  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("potential", InputFile::Required)>>chosenPotential;
  in.getOption("outputFile", InputFile::Required)>>outputFile;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
}
