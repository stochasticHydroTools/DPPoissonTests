/* Raul P. Pelaez 2020. Doubly Periodic Poisson python bindings
   Allows to call the DPPoisson module from python to compute the forces acting on a group of charges.
   For additional info use:
   import uammd
   help(uammd)

 */
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <uammd.cuh>
#include <Interactor/DoublyPeriodic/DPPoissonSlab.cuh>

namespace py = pybind11;
using DPPoissonSlab = uammd::DPPoissonSlab;
using Parameters = DPPoissonSlab::Parameters;

struct Real3ToReal4{
  __host__ __device__ uammd::real4 operator()(uammd::real3 i){
    auto pr4 = uammd::make_real4(i);
    return pr4;
  }
};
struct Real4ToReal3{
  __host__ __device__ uammd::real3 operator()(uammd::real4 i){
    auto pr3 = uammd::make_real3(i);
    return pr3;
  }
};

struct UAMMD {
  using real = uammd::real;
  std::shared_ptr<uammd::ParticleData> pd;
  std::shared_ptr<DPPoissonSlab> dppoisson;
  thrust::device_vector<uammd::real3> tmp;
  int numberParticles;
  cudaStream_t st;
  UAMMD(Parameters par, int numberParticles): numberParticles(numberParticles){
    this->pd = std::make_shared<uammd::ParticleData>(numberParticles);
    auto pg = std::make_shared<uammd::ParticleGroup>(pd, "All");
    this->dppoisson = std::make_shared<DPPoissonSlab>(pd, par);
    tmp.resize(numberParticles);
    CudaSafeCall(cudaStreamCreate(&st));
  }

  void computeForce(py::array_t<real> h_pos, py::array_t<real> h_charges, py::array_t<real> h_forces){
    {
      auto pos = pd->getPos(uammd::access::location::gpu, uammd::access::mode::write);
      thrust::copy((uammd::real3*)h_pos.data(),
		   (uammd::real3*)h_pos.data() + numberParticles,
		   tmp.begin());
      thrust::transform(thrust::cuda::par.on(st),
			tmp.begin(), tmp.end(), pos.begin(), Real3ToReal4());
      auto charges = pd->getCharge(uammd::access::location::gpu, uammd::access::mode::write);
      
      thrust::copy((uammd::real*)h_charges.data(),
		   (uammd::real*)h_charges.data() + numberParticles,
		   thrust::device_ptr<real>(charges.begin()));
      auto force = pd->getForce(uammd::access::location::gpu, uammd::access::mode::write);
      thrust::fill(thrust::cuda::par.on(st),force.begin(), force.end(), uammd::real4());
    }
    dppoisson->sum({true, false, false}, st);
    auto force = pd->getForce(uammd::access::location::gpu, uammd::access::mode::read);
    thrust::transform(thrust::cuda::par.on(st),
		      force.begin(), force.end(), tmp.begin(), Real4ToReal3());
    thrust::copy(tmp.begin(), tmp.end(), (uammd::real3*)h_forces.mutable_data());
  }
  
  ~UAMMD(){
    cudaDeviceSynchronize();
    cudaStreamDestroy(st);
  }
};



using namespace pybind11::literals;

PYBIND11_MODULE(uammd, m) {
  m.doc() = "UAMMD Doubly Periodic Poisson Python interface";
  py::class_<UAMMD>(m, "DPPoisson", "The Doubly Periodic Poisson UAMMD module").
    def(py::init<Parameters, int>(),"Parameters"_a, "numberParticles"_a).
    def("computeForce", &UAMMD::computeForce, "Computes the force acting on a group of charges due to their electrostatic interaction",
	"positions"_a,"charges"_a,"forces"_a);
  
  py::class_<DPPoissonSlab::Permitivity>(m, "Permittivity", "Permittivity in the three domains").
    def(py::init([](uammd::real top, uammd::real inside, uammd::real bottom) {
      return std::unique_ptr<DPPoissonSlab::Permitivity>(new DPPoissonSlab::Permitivity(
							{.top=top, .bottom=bottom, .inside=inside}
										       ));
    }),"top"_a = 1.0, "inside"_a = 1.0, "bottom"_a = 1.0).
    def_readwrite("top", &DPPoissonSlab::Permitivity::top).
    def_readwrite("bottom", &DPPoissonSlab::Permitivity::bottom).
    def_readwrite("inside", &DPPoissonSlab::Permitivity::inside).
    def("__str__", [](const DPPoissonSlab::Permitivity &p){
      return "permittivity = Top: " + std::to_string(p.top)+
	" Inside: "+ std::to_string(p.inside)+
	" Bottom: "+ std::to_string(p.bottom)+ "\n";
    });

  py::class_<Parameters>(m, "DPPoissonParameters", "Parameters for the Doubly Periodic Poisson module").
    def(py::init([](uammd::real Lxy,
		    uammd::real H,
		    DPPoissonSlab::Permitivity permitivity,
		    uammd::real gw,
		    uammd::real tolerance,
		    uammd::real upsampling,
		    uammd::real numberStandardDeviations,
		    int support,
		    int Nxy,
		    uammd::real split) {
      auto tmp = std::unique_ptr<Parameters>(new Parameters);
      tmp->Lxy = uammd::make_real2(Lxy, Lxy);
      tmp->H = H;
      tmp->permitivity = permitivity;
      tmp->gw=gw;
      if(split) tmp->split = split;
      if(Nxy) tmp->Nxy = Nxy;
      if(tolerance>0) tmp->tolerance = tolerance;
      if(upsampling > 0)tmp->upsampling = upsampling;
      if(numberStandardDeviations>0)tmp->numberStandardDeviations = numberStandardDeviations;
      if(support>0)tmp->support = support;
      return tmp;
    }),"Lxy"_a,
	"H"_a,
	"permittivity"_a = DPPoissonSlab::Permitivity(),
	"gw"_a,
	"tolerance"_a = -1.0,
	"upsampling"_a = -1.0,"numberStandardDeviations"_a = -1.0,"support"_a=-1,
	"Nxy"_a=-1, "split"_a=-1.0, "You must specify only either Nxy or split").
    def_readwrite("Lxy", &Parameters::Lxy).
    def_readwrite("H", &Parameters::H).
    def_readwrite("gw", &Parameters::gw).
    def_readwrite("permittivity", &Parameters::permitivity).
    def_readwrite("tolerance", &Parameters::tolerance).
    def_readwrite("upsampling", &Parameters::upsampling).
    def_readwrite("numberStandardDeviations", &Parameters::numberStandardDeviations).
    def_readwrite("support", &Parameters::support).
    def_readwrite("Nxy", &Parameters::Nxy).
    def("__str__", [](const Parameters &p){
      return "Lxy = "+std::to_string(p.Lxy.x)+"\n"+
	"H = " + std::to_string(p.H) +"\n"+
	"gw = " + std::to_string(p.gw)+ "\n" +
	"permitivity = Top: " + std::to_string(p.permitivity.top)+
	" Inside: "+ std::to_string(p.permitivity.inside)+
	" Bottom: "+ std::to_string(p.permitivity.bottom)+ "\n" +
	"tolerance = " + std::to_string(p.tolerance)+ "\n" +
	"upsampling = " + std::to_string(p.upsampling)+ "\n" +	
	"numberStandardDeviations = " + std::to_string(p.numberStandardDeviations)+ "\n" +
	"support = " + std::to_string(p.support)+ "\n" +
	"split = " + std::to_string(p.split)+ "\n" +
	"Nxy = " + std::to_string(p.Nxy)+ "\n";
    });
    
    }
