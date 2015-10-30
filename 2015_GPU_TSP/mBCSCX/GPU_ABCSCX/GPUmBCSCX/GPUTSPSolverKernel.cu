
#include "GPUTSPSolverKernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>


__device__ unsigned int randi(unsigned int seed, unsigned int min, unsigned int max) {
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	/* curand works like rand - except that it takes a state as a parameter */
	printf("<%u %u>", min, max);
	return min + ( curand(&state) % (min-max + 1) );
}

__device__ int randf(unsigned int seed, float min, float max) {

}

// gene initialization
void d_geneInit(int threadsPerBlock, int blocksPerGrid, unsigned int seed, unsigned int numCities, int *gene, const float *cityLoc) {
	printf("gene initialization at device started....\n");
	d_geneInitKernel << < threadsPerBlock, blocksPerGrid >> >(seed, numCities, gene, cityLoc);
	printf("gene initialization at device done\n");
}

__global__ void d_geneInitKernel(unsigned int seed, unsigned int numCities, int *gene, const float *cityLoc) {
	// tId: gene idx ( tId-th gene with numCities elements)
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < numCities; i++) {
		gene[tId * 2 + i] = i;	
	}
	for (int i = 1; i < numCities; i++) {
		int rIdx = randi(seed, (unsigned int) 1, (unsigned int) numCities - 1);
		int t = gene[tId * 2 + i];
		//gene[tId * 2 + i] = gene[tId * 2 + rIdx];
		//gene[tId * 2 + rIdx] = t;		
		printf("(%d[%ud (%d, %d)]", seed, rIdx, 1, numCities-1);
	}
	
	/*
	for (int i = 0; i<nPopulation; i++) {
		gene[i] = new int[nCities];
	}
	for (int i = 0; i<nPopulation; i++) {
		for (int j = 0; j<nCities; j++) gene[i][j] = j;
	}
	for (int i = 0; i<nPopulation; i++) {
		shuffleGene(i, nCities / 2);
	}*/
}


// fitness computation
void d_computeFitness(int blocksPerGrid, int threadsPerBlock, unsigned int size, const float *a, const float *b, float *c) {
	d_computeFitnessKernel << < blocksPerGrid, threadsPerBlock >> > (a, b, c);
}


__global__ void d_computeFitnessKernel(const float *a, const float *b, float *c)
{
	
}

