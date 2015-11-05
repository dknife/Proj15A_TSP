#ifndef _GPUTSPKERNEL__H_2015_YMKANG__
#define _GPUTSPKERNEL__H_2015_YMKANG__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


void safeCuda(cudaError work, const char *msg);

/////////////////// random functions
void d_initRandom(unsigned long seed, curandState *pState);
__global__ void d_initRandomKernel(unsigned long seed, curandState *pState);
__device__ unsigned int randi(curandState *pState, int add, int min, int max);
__device__ float randf(curandState *pState, int add, float min, float max);


////////////////// gene initialization
void d_geneInit(int threadsPerBlock, int blocksPerGrid, curandState *pState, unsigned int nPopulation, unsigned int numCities, int *gene, const float *cityLoc);
__global__ void d_geneInitKernel(curandState_t *state, unsigned int nPopulation, unsigned int numCities, int *gene, const float *cityLoc);


///////////////// computing fitness

// compute fitness of a specific (idx) gene
void d_computeFitnessOf(int threadsPerBlock, int blocksPerGrid, int idx, float *cityLoc, int *gene, int nPopulation, int nCities, int* d_fitness, int *distance);
__global__ void d_fitnessOf(int idx, float *cityLoc, int *gene, int nPopulation, int nCities, int *d_fitness, int *distance);

// compute the fitness values of all genes
void d_computeFitnessAll(int threadsPerBlock, int blocksPerGrid, float *cityLoc, int *gene, int nPopulation, int nCities, int* d_fitness);
__global__ void d_computeAllFitnessKernel(float *cityLoc, int *gene, int nPopulation, int nCities, int *d_fitness);

// gene copy
void d_copyGene(int threadsPerBlock, int blocksPerGrid, int toIdx, int fromIdx, int *d_gene, int nCities);
__global__ void d_copyGeneKernel(int toIdx, int fromIdx, int *d_gene, int nCities);

// gene crossover
void d_initAuxMem(int THREADSPERBLOCK, int blocksPerGrid, int nCities, int i, int *d_gene, int *d_orderOfCity, int *d_fJump, int *d_bJump);
__global__ void d_initAuxMemKernel(int nCities, int i, int *d_gene, int *d_orderOfCity, int *d_fJump, int *d_bJump);

void d_crossover(int THREADSPERBLOCK, int blocksPerGrid, int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity, int *d_fJump, int *d_bJump);
__global__ void d_crossoverKernel(int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity, int *d_fJump, int *d_bJump);

void d_crossoverABCSCX(int THREADSPERBLOCK, int blocksPerGrid, int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity, int *d_fJump, int *d_bJump);
__global__ void d_crossoverABCSCXKernel(int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity, int *d_fJump, int *d_bJump);

// gene mutate
void d_mutateGene(int threadsPerBlock, int blocksPerGrid, int gene_idx, int idxA, int idxB, int *d_gene, int nCities);
__global__ void d_mutateGeneKernel(int gene_idx, int idxA, int idxB, int *d_gene, int nCities);



#endif