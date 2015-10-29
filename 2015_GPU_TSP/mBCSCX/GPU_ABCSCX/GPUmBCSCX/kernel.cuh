#ifndef _KERNEL_H_2015_YMKANG_MBCSCXGPU_HH_
#define _KERNEL_H_2015_YMKANG_MBCSCXGPU_HH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void callKernelByHost(int blocksPerGrid, int threadsPerBlock, unsigned int size, const float *a, const float *b, float *c);

__host__ void callKernel(unsigned int size, float *c, const float *a, const float *b);

__global__ void addKernel(const float *a, const float *b, float *c);

#endif