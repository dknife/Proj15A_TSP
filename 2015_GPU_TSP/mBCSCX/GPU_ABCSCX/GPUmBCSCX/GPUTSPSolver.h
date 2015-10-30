#ifndef _GPU_TSP_SOLVER
#define _GPU_TSP_SOLVER

#ifdef WIN32 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#endif

#include "GeneticTSPSolver.h"
#include "GPUTSPSolverKernel.cuh"

class CGPUTSPSolver : public CGeneticTSPSolver {
	// host memory
	float *h_CityLoc;

	// device memory
	float *d_CityLoc;
	int   *d_fitness, *d_gene;

	// cuda works
	void CleanCudaMemory(void);
	void PrepareCudaMemory(void);
	void MoveCityLocToCudaMemory(void);
	void GeneInitCudaMemory(void);

public:
	CGPUTSPSolver(CCityLocData *inputData, int nGenes, int nGroups);
	CGPUTSPSolver();
	~CGPUTSPSolver();

	void LoadData(CCityLocData *inputData, int nGenes, int nGroups);
	void initSolver(void);

	// GPU version should be made
	// fixGene
	// computeFitness
	// nextGeneration

	void cudaTestFunction(void);
};
#endif