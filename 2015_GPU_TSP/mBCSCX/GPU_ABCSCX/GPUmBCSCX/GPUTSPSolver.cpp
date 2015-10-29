
#include "GPUTSPSolver.h"
#include "kernel.cuh"


void safeCuda(cudaError work) { if (work != cudaSuccess) exit(EXIT_FAILURE); }


CGPUTSPSolver::CGPUTSPSolver() {
	d_CityLoc = d_fitness = d_gene = NULL;
}

CGPUTSPSolver::~CGPUTSPSolver() {
	
}

void CGPUTSPSolver::PrepareCudaMemory(void) {
	// create these data within CudaDevice
	// cityLoc : int [nCities] 
	// fitness : int [nPopulation]
	// genes   : int [nPopulation][nCities]

	cudaError_t err = cudaSuccess;
	int nCities = CGeneticTSPSolver::getNumCities();
	int nPopulation = CGeneticTSPSolver::getNumPopulation();


	safeCuda(cudaMalloc((void **)&d_CityLoc, nCities * sizeof(int)));
	safeCuda(cudaMalloc((void **)&d_fitness, nPopulation * sizeof(int)));
	safeCuda(cudaMalloc((void **)&d_gene, nPopulation * nCities * sizeof(int)));


	printf("cuda device memory successfully allocated (%d cities, %d genes)\n", nCities, nPopulation);

	//safeCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	//safeCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	//int threadsPerBlock = 256;
	//int blocksPerGrid = (nPopulation + threadsPerBlock - 1) / threadsPerBlock;
	//printf("cuda kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	//safeCuda(cudaGetLastError());
}

void CGPUTSPSolver::LoadData(CCityLocData *inputData, int nGenes, int nGroups) {

	CGeneticTSPSolver::LoadData(inputData, nGenes, nGroups);

	PrepareCudaMemory();
		
	
}

void CGPUTSPSolver::cudaTestFunction(void) {
	cudaError_t err = cudaSuccess;
	int numElements = 10;
	size_t size = numElements * sizeof(float);
	
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	if (!h_A || !h_B || !h_C) exit(EXIT_FAILURE);

	for (int i = 0; i < numElements; ++i) {
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	float *d_A, *d_B, *d_C;
	d_A = d_B = d_C = NULL;

	safeCuda(cudaMalloc((void **)&d_A, size));	
	safeCuda(cudaMalloc((void **)&d_B, size));
	safeCuda(cudaMalloc((void **)&d_C, size));
	safeCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	safeCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("cuda kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	callKernelByHost(blocksPerGrid, threadsPerBlock, numElements, d_A, d_B, d_C);
	safeCuda(cudaGetLastError());

	safeCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
	safeCuda(cudaFree(d_A));
	safeCuda(cudaFree(d_B));
	safeCuda(cudaFree(d_C));

	for (int i = 0; i < numElements; i++) {
		printf("%f + %f = %f (%f)\n", h_A[i], h_B[i], h_C[i], h_A[i] + h_B[i]);
	}

	free(h_A);
	free(h_B);
	free(h_C);

	safeCuda(cudaDeviceReset());
}