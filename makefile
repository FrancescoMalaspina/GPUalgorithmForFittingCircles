# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects
# NB 	%.cu : 	CUDA file
#	%.cxx:	ROOT file
#	%.cpp:	c++ normal file
#	%.cu :	CUDA file

.PHONY = all clean	#define all the targets that are not files

# variabili per CUDA
LINKERFLAG = --Wno-deprecated-gpu-targets --gpu-architecture sm_50
SRCS := $(wildcard *.cu)
BINS := $(SRCS:%.cu=%)

# variabili per ROOT
R_LINKERFLAG = -std=c++11 -I$(shell root-config --incdir)
R_SRCS := $(wildcard *.cxx)
R_BINS := $(R_SRCS:%.cxx=%)
LIB = $(shell root-config --libs )

# varibili per file c++
N_SRCS := $(wildcard *.cpp)
N_BINS := $(N_SRCS:%.cpp=%)


all: ${BINS} ${R_BINS} ${N_BINS} ${U_BINS}

%: %.cu
	nvcc  ${LINKERFLAG} -o $@ $<

%: %.cxx
	g++ ${R_LINKERFLAG} -o $@ $< ${LIB}

%: %.cpp
	g++ -o $@ $<

clean:
	@echo "Cleaning up..."
	rm -rvf *.o ${BINS} ${R_BINS} ${N_BINS}
	rm fit.dat simul.dat
