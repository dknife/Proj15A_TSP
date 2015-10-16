
#include "kernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__host__ void callKernel(unsigned int size, int *c, const int *a, const int *b) {
	addKernel <<< 1, size >>> (c, a, b);
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

