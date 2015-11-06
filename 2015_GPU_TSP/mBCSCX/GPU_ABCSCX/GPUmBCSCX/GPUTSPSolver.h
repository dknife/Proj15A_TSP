#ifndef _GPU_TSP_SOLVER
#define _GPU_TSP_SOLVER

#ifdef WIN32 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "GeneticTSPsolver.h"
#include "GPUTSPSolverKernel.cuh"

#define THREADSPERBLOCK  16

class CGPUTSPSolver : public CGeneticTSPSolver {

	int* currentBestGene;

	// host memory
	float *h_CityLoc;
	int   *h_fitness;

	// device memory
	float *d_CityLoc;
	int   *d_fitness;
	int   *d_gene; // nPopulation * nCities
	int   *d_aGene; // nCities

	//////////////////////////// distance array for a single gene
	int   *h_distanceSeq;
	int   *d_distanceSeq; // array storing distances between a node and the next node in a sequence

	// GPU memory for BCSCS/ABCSCX computation
	int    *d_fJump;  // nGenes x nCities
	int    *d_bJump;  // nGenes x nCities 
	int    *d_orderOfCity; // nGenes x nCities

	// cuda random state
	curandState *d_cudaState;

	// cuda works
	void CleanCudaMemory(void);
	void PrepareCudaMemory(void);
	void MoveCityLocToCudaMemory(void);
	void GeneInitCudaMemory(void);

	// 
	void initCuRand(unsigned long seed);
	

public:
	CGPUTSPSolver(CCityLocData *inputData, int nGenes, int nGroups);
	CGPUTSPSolver();
	~CGPUTSPSolver();

	void LoadData(CCityLocData *inputData, int nGenes, int nGroups);
	void LoadSolution(const char *fname);
	void initSolver(void);
	void nextGeneration(void);

	void computeFitnessOf(int idx);
	void computeFitness(void);

	void copySolution(int *SolutionCpy);
	void fixGene(int idx);

	// GPU version should be made
	// fixGene
	// computeFitness
	// nextGeneration

};
#endif