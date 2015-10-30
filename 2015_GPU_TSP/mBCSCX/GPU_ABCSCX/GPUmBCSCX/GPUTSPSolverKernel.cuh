#ifndef _GPUTSPKERNEL__H_2015_YMKANG__
#define _GPUTSPKERNEL__H_2015_YMKANG__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void d_geneInit(int blocksPerGrid, int threadsPerBlock , unsigned int seed, unsigned int numCities, int *gene, const float *cityLoc);
__global__ void d_geneInitKernel(unsigned int seed, unsigned int numCities, int *gene, const float *cityLoc);


void d_computeFitness(int blocksPerGrid, int threadsPerBlock, unsigned int size, const float *a, const float *b, float *c);
__global__ void d_computeFitnessKernel(const float *a, const float *b, float *c);




#endif