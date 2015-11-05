
#include "GPUTSPSolverKernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


#include <stdio.h>
#include <math.h>


void safeCuda(cudaError work, const char *msg) { if (work != cudaSuccess) { printf("CUDA ERROR at (%s) with code %d\n", msg, work);  exit(EXIT_FAILURE); } }


void d_initRandom(unsigned long seed, curandState *pState) {
	d_initRandomKernel << <1, 1 >> >(seed, pState);
}

__global__ void d_initRandomKernel(unsigned long seed, curandState *pState) {
	curand_init(seed, 0, 0, pState);
}

__device__ unsigned int randi(curandState *pState, int add, int min, int max) {
	curandState localState = *pState;
	unsigned int rndval = min + (curand(&localState) * add + add)%(max-min+1);
	*pState = localState;
	return rndval;
}

__device__ float randf(curandState *pState, int add, float min, float max) {
	curandState localState = *pState;
	float rndval = min + (((curand(&localState) * add + add)%100000)/100000.0)*(max - min);
	*pState = localState;
	return rndval;
}

// gene initialization
void d_geneInit(int threadsPerBlock, int blocksPerGrid, curandState *pState, unsigned int nPopulation, unsigned int numCities, int *gene, const float *cityLoc) {

	printf("gene initialization at device started....\n");
	d_geneInitKernel << < threadsPerBlock, blocksPerGrid >> >(pState, nPopulation, numCities, gene, cityLoc);
	printf("gene initialization at device done\n");
}

__global__ void d_geneInitKernel(curandState_t *pstate, unsigned int nPopulation, unsigned int numCities, int *gene, const float *cityLoc) {
	
	// tId: gene idx ( tId-th gene with numCities elements)
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= nPopulation) return;

	// gene initialization - cards straight
	for (int i = 0; i < numCities; i++) {
		gene[tId*numCities+i] = i;
	}
	
	// gene shuffle
	for (int i = 1; i < numCities; i++) {
		int rIdx = randi(pstate, threadIdx.x+blockIdx.x+blockDim.x,  (unsigned int) 1, (unsigned int) numCities - 1);
		int t = gene[tId*numCities + i];
		gene[tId * numCities + i] = gene[tId * numCities + rIdx];
		gene[tId * numCities + rIdx] = t;
	}
}


// compute fitness of a specific (idx) gene
void d_computeFitnessOf(int threadsPerBlock, int blocksPerGrid, int idx, float *cityLoc, int *gene, int nPopulation, int nCities, int *fitness, int *distance) {

	d_fitnessOf << < threadsPerBlock, blocksPerGrid >> > (idx, cityLoc, gene, nPopulation, nCities, fitness, distance);

}

__global__ void d_fitnessOf(int idx, float *cityLoc, int *gene, int nPopulation, int nCities, int *fitness, int *distance) {

	// tId: index within a gene
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= nCities) return;
	int idxA = gene[idx*nCities+tId];
	int idxB = gene[idx*nCities+((tId + 1) % nCities)];
	float dx = cityLoc[idxA * 2] - cityLoc[idxB * 2];
	float dy = cityLoc[idxA * 2 + 1] - cityLoc[idxB * 2 + 1];

	distance[tId] = (int)(sqrt(double(dx*dx + dy*dy) + 0.5));

	//printf("[%d, tId:%d-%d] distance %d(%5.2f, %5.2f) %d(%5.2f, %5.2f): (%5.2f, %5.2f) %d\n", idx, tId, (tId + 1) % nCities, idxA, cityLoc[idxA * 2], cityLoc[idxA * 2 + 1], idxB, cityLoc[idxB * 2], cityLoc[idxB * 2 + 1], dx, dy, distance[tId]);
}

// compute the fitness values of all genes
void d_computeFitnessAll(int threadsPerBlock, int blocksPerGrid, float *cityLoc, int *gene, int nPopulation, int nCities, int *fitness) {

	d_computeAllFitnessKernel << < threadsPerBlock, blocksPerGrid >> > (cityLoc, gene, nPopulation, nCities, fitness);
}

__global__ void d_computeAllFitnessKernel(float *cityLoc, int *gene, int nPopulation, int nCities, int *fitness) {

	
	// tId: index within a gene
	int tId = threadIdx.x + blockIdx.x * blockDim.x;	
	if (tId >= nPopulation) return;

	fitness[tId] = 0;
	
	for (int i = 0; i < nCities; i++) {
		int idxA = gene[tId*nCities + i];
		int idxB = gene[tId*nCities + ((i + 1) % nCities)];
		float dx = cityLoc[idxA * 2] - cityLoc[idxB * 2];
		float dy = cityLoc[idxA * 2 + 1] - cityLoc[idxB * 2 + 1];
		fitness[tId] = fitness[tId] + (int)(sqrt(double(dx*dx + dy*dy) + 0.5));
	}
}


// gene copy
void d_copyGene(int threadsPerBlock, int blocksPerGrid, int toIdx, int fromIdx, int *d_gene, int nCities) {
	d_copyGeneKernel << <threadsPerBlock, blocksPerGrid >> >(toIdx, fromIdx, d_gene, nCities);
}

__global__ void d_copyGeneKernel(int toIdx, int fromIdx, int *d_gene, int nCities) {
	// tId: index within a gene
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= nCities) return;

	d_gene[toIdx*nCities + tId] = d_gene[fromIdx*nCities + tId];

}

// gene crossover
void d_initAuxMem(int threadsPerBlock, int blocksPerGrid, int nCities, int i, int *d_gene, int *d_orderOfCity, int *d_fJump, int *d_bJump) {
	d_initAuxMemKernel << < threadsPerBlock, blocksPerGrid >> > (nCities, i, d_gene, d_orderOfCity, d_fJump, d_bJump);
	
}
__global__ void d_initAuxMemKernel(int nCities, int i, int *d_gene, int *d_orderOfCity, int *d_fJump, int *d_bJump) {
	// tId: index within a gene
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= nCities) return;

	d_fJump[i*nCities + tId] = (tId == nCities-1)? 2:1;
	d_bJump[i*nCities + tId] = (tId == 1)? 2: 1;

	int city = d_gene[i*nCities + tId];
	d_orderOfCity[i*nCities + city] = tId;
}

void d_crossover(int threadsPerBlock, int blocksPerGrid, int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity,  int *d_fJump, int *d_bJump) {
	d_crossoverKernel << < threadsPerBlock, blocksPerGrid >> > (i, nCrossover, nCities, d_gene, d_cityLoc, d_orderOfCity, d_fJump, d_bJump);
}
__global__ void d_crossoverKernel(int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity, int *d_fJump, int *d_bJump) {
	// tId: index within a gene
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= nCrossover ) return;

	int p1 = tId * 2;
	int p2 = tId * 2 + 1;
	int child = nCrossover*2 + tId;
	
	int lastCity = d_gene[child*nCities + i - 1];

	// find candidates from parent 1	
	int idx1 = d_orderOfCity[p1*nCities + lastCity];
	int fnode = (idx1 + d_fJump[p1*nCities + idx1]) % nCities;
	int bnode =  idx1 - d_bJump[p1*nCities + idx1];
	while (bnode < 0) bnode += nCities;
	int cand1 = d_gene[p1*nCities + fnode];
	int cand2 = d_gene[p1*nCities + bnode];

	// find candidates from parent 2	
	int idx2 = d_orderOfCity[p2*nCities + lastCity];
	fnode = (idx2 + d_fJump[p2*nCities + idx2]) % nCities;
	bnode =  idx2 - d_bJump[p2*nCities + idx2];
	while (bnode < 0) bnode += nCities;
	int cand3 = d_gene[p2*nCities + fnode];
	int cand4 = d_gene[p2*nCities + bnode];
	
	// select best candidate

	float dx = d_cityLoc[cand1 * 2]     - d_cityLoc[lastCity * 2];
	float dy = d_cityLoc[cand1 * 2 + 1] - d_cityLoc[lastCity * 2 + 1];
	int dist1 =  (int)(sqrt(double(dx*dx + dy*dy) + 0.5));

	dx = d_cityLoc[cand2 * 2] - d_cityLoc[lastCity * 2];
	dy = d_cityLoc[cand2 * 2 + 1] - d_cityLoc[lastCity * 2 + 1];
	int dist2 = (int)(sqrt(double(dx*dx + dy*dy) + 0.5));

	dx = d_cityLoc[cand3 * 2] - d_cityLoc[lastCity * 2];
	dy = d_cityLoc[cand3 * 2 + 1] - d_cityLoc[lastCity * 2 + 1];
	int dist3 = (int)(sqrt(double(dx*dx + dy*dy) + 0.5));

	dx = d_cityLoc[cand4 * 2] - d_cityLoc[lastCity * 2];
	dy = d_cityLoc[cand4 * 2 + 1] - d_cityLoc[lastCity * 2 + 1];
	int dist4 = (int)(sqrt(double(dx*dx + dy*dy) + 0.5));

	int best = dist1;
	int bestCity = cand1;
	if (dist2 < best) {	best = dist2;  bestCity = cand2; }
	if (dist3 < best) { best = dist3;  bestCity = cand3; }
	if (dist4 < best) { best = dist4;  bestCity = cand4; }
	
	// set the bestCity as the next element of the offspring
	//printf("bestCity = %d\n", bestCity);
	d_gene[child*nCities + i] = bestCity;

	// invalidate nextCity;
	idx1 = d_orderOfCity[p1*nCities + bestCity];
	int fChange = idx1 - d_bJump[p1*nCities + idx1];
	while (fChange < 0) fChange += nCities;
	int bChange = (idx1 + d_fJump[p1*nCities + idx1]) % nCities;
	d_fJump[p1*nCities + fChange] += d_fJump[p1*nCities + idx1];
	d_bJump[p1*nCities + bChange] += d_bJump[p1*nCities + idx1];

	idx2 = d_orderOfCity[p2*nCities + bestCity];
	fChange = idx2 - d_bJump[p2*nCities + idx2];
	while (fChange < 0) fChange += nCities;
	bChange = (idx2 + d_fJump[p2*nCities + idx2]) % nCities;
	d_fJump[p2*nCities + fChange] += d_fJump[p2*nCities + idx2];
	d_bJump[p2*nCities + bChange] += d_bJump[p2*nCities + idx2];
}

void d_crossoverABCSCX(int threadsPerBlock, int blocksPerGrid, int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity, int *d_fJump, int *d_bJump) {
	d_crossoverKernel << < threadsPerBlock, blocksPerGrid >> > (i, nCrossover, nCities, d_gene, d_cityLoc, d_orderOfCity, d_fJump, d_bJump);
}
__global__ void d_crossoverABCSCXKernel(int i, int nCrossover, int nCities, int *d_gene, float *d_cityLoc, int *d_orderOfCity, int *d_fJump, int *d_bJump) {
	// tId: index within a gene
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= nCrossover) return;

	int p1 = tId * 2;
	int p2 = p1 + 1;
	int p = (i%2)? p1:p2;
	int child = nCrossover * 2 + tId;

	int lastCity = d_gene[child*nCities + i - 1];

	// find candidates from parent	
	int idx = d_orderOfCity[p*nCities + lastCity];
	int fnode = (idx + d_fJump[p*nCities + idx]) % nCities;
	int bnode = idx - d_bJump[p*nCities + idx];
	while (bnode < 0) bnode += nCities;
	int cand1 = d_gene[p*nCities + fnode];
	int cand2 = d_gene[p*nCities + bnode];

	// select better candidate

	float dx = d_cityLoc[cand1 * 2] - d_cityLoc[lastCity * 2];
	float dy = d_cityLoc[cand1 * 2 + 1] - d_cityLoc[lastCity * 2 + 1];
	int dist1 = (int)(sqrt(double(dx*dx + dy*dy) + 0.5));

	dx = d_cityLoc[cand2 * 2] - d_cityLoc[lastCity * 2];
	dy = d_cityLoc[cand2 * 2 + 1] - d_cityLoc[lastCity * 2 + 1];
	int dist2 = (int)(sqrt(double(dx*dx + dy*dy) + 0.5));

	int best = dist1;
	int bestCity = cand1;
	if (dist2 < best) { best = dist2;  bestCity = cand2; }
	
	// set the bestCity as the next element of the offspring
	//printf("bestCity = %d\n", bestCity);
	d_gene[child*nCities + i] = bestCity;

	// invalidate nextCity;
	int idx1 = d_orderOfCity[p1*nCities + bestCity];
	int fChange = idx1 - d_bJump[p1*nCities + idx1];
	while (fChange < 0) fChange += nCities;
	int bChange = (idx1 + d_fJump[p1*nCities + idx1]) % nCities;
	d_fJump[p1*nCities + fChange] += d_fJump[p1*nCities + idx1];
	d_bJump[p1*nCities + bChange] += d_bJump[p1*nCities + idx1];

	int idx2 = d_orderOfCity[p2*nCities + bestCity];
	fChange = idx2 - d_bJump[p2*nCities + idx2];
	while (fChange < 0) fChange += nCities;
	bChange = (idx2 + d_fJump[p2*nCities + idx2]) % nCities;
	d_fJump[p2*nCities + fChange] += d_fJump[p2*nCities + idx2];
	d_bJump[p2*nCities + bChange] += d_bJump[p2*nCities + idx2];
}


// gene(gene_idx) mutate : revers the substring from idxA to idxB
void d_mutateGene(int threadsPerBlock, int blocksPerGrid, int gene_idx, int idxA, int idxB, int *d_gene, int nCities) {
	d_mutateGeneKernel <<< threadsPerBlock, blocksPerGrid>>> (gene_idx, idxA, idxB, d_gene, nCities);
}
__global__ void d_mutateGeneKernel(int gene_idx, int idxA, int idxB, int *d_gene, int nCities) {

	int half = (idxB - idxA + 1) / 2;
	// tId: index within a gene
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= half) return;

	int t = d_gene[gene_idx*nCities + tId + idxA];
	d_gene[gene_idx*nCities + idxA + tId] = d_gene[gene_idx*nCities + idxB - tId];
	d_gene[gene_idx*nCities + idxB - tId] = t;

}



