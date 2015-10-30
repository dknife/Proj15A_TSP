
#include "GPUTSPSolver.h"
#include "kernel.cuh"


void safeCuda(cudaError work) { if (work != cudaSuccess) { printf("CUDA ERROR\n");  exit(EXIT_FAILURE); } }


CGPUTSPSolver::CGPUTSPSolver() {
	h_CityLoc = NULL;

	d_CityLoc = NULL;
	d_fitness = d_gene = NULL;
}

CGPUTSPSolver::~CGPUTSPSolver() {
	
}

void CGPUTSPSolver::CleanCudaMemory(void) {

	printf("cleaning cuda memories\n");
	if (h_CityLoc) free(h_CityLoc);
	
	if (d_CityLoc) safeCuda(cudaFree(d_CityLoc));
	if (d_fitness) safeCuda(cudaFree(d_fitness));
	if (d_gene)    safeCuda(cudaFree(d_gene));

}

void CGPUTSPSolver::PrepareCudaMemory(void) {
	// create these data within CudaDevice
	// cityLoc : int [nCities] 
	// fitness : int [nPopulation]
	// genes   : int [nPopulation][nCities]

	printf("preparing cuda memories...\n");
	cudaError_t err = cudaSuccess;


	h_CityLoc = (float *)malloc(nCities * 2 * sizeof(float));
	safeCuda(cudaMalloc((void **)&d_CityLoc, nCities * 2 * sizeof(float)));
	
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

void CGPUTSPSolver::MoveCityLocToCudaMemory(void) {
	
	printf("move city location data to cuda memory\n");
	CCityLocData *city = CGeneticTSPSolver::getCityLocData();
	
	for (int i = 0; i < nCities; i++) {
		Point p = city->getLocation(i);
		h_CityLoc[i * 2] = p.x;
		h_CityLoc[i * 2 + 1] = p.y;
	}

}

void CGPUTSPSolver::GeneInitCudaMemory(void) {

	int threadsPerBlock = 512;
	int blocksPerGrid = (nPopulation + threadsPerBlock - 1) / threadsPerBlock;
	
	if (!d_gene) printf("no gene memory\n");

	printf("gene initialization with %d threads per block x (%d blocks)\n", threadsPerBlock, blocksPerGrid);

	d_geneInit(threadsPerBlock, blocksPerGrid, time(NULL), nCities, d_gene, d_CityLoc);

	///////////////////// should be deleted 
	for (int i = 0; i<nPopulation; i++) {
		gene[i] = new int[nCities];
	}
	for (int i = 0; i<nPopulation; i++) {
		for (int j = 0; j<nCities; j++) gene[i][j] = j;
	}
	for (int i = 0; i<nPopulation; i++) {
		shuffleGene(i, nCities / 2);
	}
	/////////////////////////////////////////////////////

	//computeFitness();
	
	printf("initSolver GPU\n");

	
}

void CGPUTSPSolver::LoadData(CCityLocData *inputData, int nGenes, int nGroups) {

	CGeneticTSPSolver::LoadData(inputData, nGenes, nGroups);

	printf("LoadData - GPU\n");
	CleanCudaMemory();
	PrepareCudaMemory();
	MoveCityLocToCudaMemory();	
	
}

void CGPUTSPSolver::initSolver(void) {

	nGeneration = 0;
	Temperature = 100.0;
	crossoverMethod = CROSSOVERMETHOD::BCSCX;
	recordBroken = false;
	bHeating = false;

	GeneInitCudaMemory();

	printf("gene data at cuda memory generated\n");
	
	
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