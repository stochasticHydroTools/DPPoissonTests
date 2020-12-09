
NVCC=nvcc
#You might have to change this if you want to use MKL instead of lapacke. Only one is needed.
LAPACKE_FLAGS=-llapacke -I/usr/include/lapacke
#MKL_FLAGS=-DUSE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64  -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

UAMMD_ROOT=source/uammd/
INCLUDEFLAGS= -I $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party 
BASIC_LINE= $(NVCC) -O3 -std=c++11 -x cu $(INCLUDEFLAGS) --expt-relaxed-constexpr $(MKL_FLAGS) $(LAPACKE_FLAGS)


all: poisson python

poisson: source/PoissonSlab.cu source/RepulsivePotential.cuh
	$(BASIC_LINE) source/PoissonSlab.cu  -o poisson -lcufft

python:
	make -C python_interface
test: poisson
	(cd charged_wall; bash test.bash);
	(cd uncharged_wall; bash test.bash);
	(cd rdftest; bash testRDF.bash);
clean:
	make -C python_interface clean
	rm -rf charged_wall/results
	rm -rf uncharged_wall/results
	rm -rf rdftest/results
	rm -f poisson





