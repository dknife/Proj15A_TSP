#ifndef _GPU_TSP_SOLVER
#define _GPU_TSP_SOLVER

#ifdef WIN32 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#endif

#include "GeneticTSPSolver.h"

class CGPUTSPSolver : public CGeneticTSPSolver {
	// device memory
	int *d_CityLoc, *d_fitness, *d_gene;

	void PrepareCudaMemory(void);
public:
	CGPUTSPSolver(CCityLocData *inputData, int nGenes, int nGroups);
	CGPUTSPSolver();
	~CGPUTSPSolver();

	void LoadData(CCityLocData *inputData, int nGenes, int nGroups);

	// GPU version should be made
	// fixGene
	// computeFitness
	// nextGeneration

	void cudaTestFunction(void);
};
#endif