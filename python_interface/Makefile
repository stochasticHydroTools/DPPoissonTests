
UAMMD_ROOT=../source/uammd/
PYBIND_ROOT=pybind11/
PYTHON=python3
NVCC=nvcc
#You can replace lapacke and cblas by intel's MKL using -DUSE_MKL and linking with that instead
LAPACKE_FLAGS=-llapacke -I/usr/include/lapacke
#MKL_FLAGS=-DUSE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64  -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

#UAMMD can be quite verbose, 5 shows only some messages at initialization/exit, 0 will only print critical errors, 15will print A LOT.
VERBOSITY=0
#Uncomment for double precision, UAMMD is compiled in single by default
#DOUBLEPRECISION=-DDOUBLE_PRECISION
#In caso you prefer to import with other name
MODULE_NAME=uammd
INCLUDE_FLAGS= -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/


LIBRARY_NAME=$(MODULE_NAME)`$(PYTHON)-config --extension-suffix`
FILE=uammd_wrapper.cu
all:
	$(NVCC) -w -shared -std=c++11 -DMAXLOGLEVEL=$(VERBOSITY) $(DOUBLEPRECISION) $(INCLUDE_FLAGS) -Xcompiler "-fPIC -w"  $(FILE) -o $(LIBRARY_NAME) $(LAPACKE_FLAGS) $(MKL_FLAGS) -lcufft
clean:
	rm $(LIBRARY_NAME)
