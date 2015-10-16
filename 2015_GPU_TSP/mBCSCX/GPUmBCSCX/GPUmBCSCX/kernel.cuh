#ifndef _KERNEL_H_2015_YMKANG_MBCSCXGPU_HH_
#define _KERNEL_H_2015_YMKANG_MBCSCXGPU_HH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ void callKernel(unsigned int size, int *c, const int *a, const int *b);

__global__ void addKernel(int *c, const int *a, const int *b);

#endif