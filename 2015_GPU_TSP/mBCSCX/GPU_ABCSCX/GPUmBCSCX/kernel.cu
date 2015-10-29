
#include "kernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


void callKernelByHost(int blocksPerGrid, int threadsPerBlock, unsigned int size, const float *a, const float *b, float *c) {
	addKernel <<< blocksPerGrid, threadsPerBlock >>> (a, b, c);
}

//__host__ void callKernel(unsigned int size, int *c, const int *a, const int *b) {
//	addKernel <<< 1, size >>> (c, a, b);
//}

__global__ void addKernel(const float *a, const float *b, float *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

