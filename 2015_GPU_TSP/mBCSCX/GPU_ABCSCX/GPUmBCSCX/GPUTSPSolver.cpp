
#include "GPUTSPSolver.h"
#include <stdio.h>
#include <stdlib.h>





CGPUTSPSolver::CGPUTSPSolver() {

	currentBestGene = NULL;
	THREADSPERBLOCK = 256;

	h_CityLoc = NULL;
	d_CityLoc = NULL;

	h_fitness = NULL;
	d_fitness = NULL;

	d_gene  = NULL;
	d_aGene = NULL;

	h_distanceSeq = NULL;
	d_distanceSeq = NULL;

	d_cudaState = NULL;

	d_orderOfCity = d_fJump = d_bJump = NULL;

	for(int i=0;i<100;i++) localMinima[i]= NULL;
	nLocalMin = 0;
}

CGPUTSPSolver::~CGPUTSPSolver() {
	
}

void CGPUTSPSolver::CleanCudaMemory(void) {

	printf("cleaning cuda memories\n");

	if (currentBestGene) free(currentBestGene);

	if (h_CityLoc) free(h_CityLoc);
	if (h_fitness) free(h_fitness);

	if (h_distanceSeq) free(h_distanceSeq);


	if (d_CityLoc)   safeCuda(cudaFree(d_CityLoc), "free d_CityLoc");
	if (d_fitness)   safeCuda(cudaFree(d_fitness), "free d_fitness");

	if (d_gene)      safeCuda(cudaFree(d_gene), "free d_gene");
	if (d_aGene)     safeCuda(cudaFree(d_aGene), "free d_aGene");


	if (d_distanceSeq)    safeCuda(cudaFree(d_distanceSeq), "free d_distanceSeq");
	if (d_cudaState) safeCuda(cudaFree(d_cudaState), "free d_cudaState");


	// GPU memory for crossover
	if (d_orderOfCity)    safeCuda(cudaFree(d_orderOfCity), "free d_orderOfCity");
	if (d_fJump)    safeCuda(cudaFree(d_fJump), "free d_fJump");
	if (d_bJump)    safeCuda(cudaFree(d_bJump), "free d_bJump");

	for (int i = 0; i < 100; i++) {
		if (localMinima[i]) delete[] localMinima[i]; localMinima[i] = NULL;
	}
	nLocalMin = 0;
	
}


void CGPUTSPSolver::PrepareCudaMemory(void) {
	// create these data within CudaDevice
	// cityLoc : int [nCities] 
	// fitness : int [nPopulation]
	// genes   : int [nPopulation][nCities]

	printf("preparing cuda memories...\n");
	cudaError_t err = cudaSuccess;


	currentBestGene = (int *)  malloc(nCities     * sizeof(int));
	h_CityLoc       = (float *)malloc(nCities * 2 * sizeof(float));
	h_fitness       = (int *)  malloc(nPopulation * sizeof(int));
	h_distanceSeq   = (int *)  malloc(nCities     * sizeof(int));

	for (int i = 0; i < 128; i++) {
		localMinima[i] = (int *)malloc(nCities * sizeof(int));
	}
	nLocalMin = 0;

	// device multi-element array
	safeCuda(cudaMalloc((void **)&d_fitness, nPopulation * sizeof(int)), "alloc d_fitness");
	safeCuda(cudaMalloc((void **)&d_CityLoc,      nCities * 2 * sizeof(float)), "alloc d_CityLoc");	
	safeCuda(cudaMalloc((void **)&d_distanceSeq,  nCities     * sizeof(int)), "alloc d_distanceSeq");

	safeCuda(cudaMalloc((void **)&d_gene , nPopulation * nCities * sizeof(int)), "alloc d_gene");
	safeCuda(cudaMalloc((void **)&d_aGene, nCities * sizeof(int)), "alloc d_aGene");
	
	// GPU memory for crossover
	safeCuda(cudaMalloc((void **)&d_orderOfCity, nPopulation * nCities * sizeof(int)), "alloc d_orderOfCity");
	safeCuda(cudaMalloc((void **)&d_fJump, nPopulation * nCities * sizeof(int)), "alloc d_fJump");
	safeCuda(cudaMalloc((void **)&d_bJump, nPopulation * nCities * sizeof(int)), "alloc d_bJump");

	// device 1-element data
	safeCuda(cudaMalloc((void **)&d_cudaState, sizeof(curandState)), "alloc cudaState");

	printf("cuda device memory successfully allocated (%d cities, %d genes)\n", nCities, nPopulation);


}

void CGPUTSPSolver::MoveCityLocToCudaMemory(void) {
	
	printf("move city location data to cuda memory\n");
	CCityLocData *city = CGeneticTSPSolver::getCityLocData();
	
	for (int i = 0; i < nCities; i++) {
		Point p = city->getLocation(i);
		h_CityLoc[i * 2] = p.x;
		h_CityLoc[i * 2 + 1] = p.y;
		//printf("%d(%5.2f, %5.2f)\n", i, p.x, p.y);
	}

	safeCuda(cudaMemcpy(d_CityLoc, h_CityLoc, nCities * 2 * sizeof(float), cudaMemcpyHostToDevice), "device-to-host memory copy: h_CityLoc -> d_CityLoc" );
}

void CGPUTSPSolver::initCuRand(unsigned long seed) {
	d_initRandom(time(NULL), this->d_cudaState);
}

void CGPUTSPSolver::GeneInitCudaMemory(void) {

	int blocksPerGrid = (nPopulation + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	
	if (!d_gene) printf("no gene memory\n");

	printf("gene initialization with %d threads per block x (%d blocks)\n", THREADSPERBLOCK, blocksPerGrid);

	d_geneInit(blocksPerGrid, THREADSPERBLOCK, d_cudaState, nPopulation, nCities, d_gene, d_CityLoc);

	//safeCuda(cudaGetLastError(), "Get Last Error at CGPUTSPSolver::GeneInitCudaMemory");


	
	printf("initSolver GPU\n");

	
}

void CGPUTSPSolver::LoadData(CCityLocData *inputData, int nGenes, int nGroups) {

	printf("Loading data with nGenes = %d\n", nGenes);
	CGeneticTSPSolver::LoadData(inputData, nGenes, nGroups);

	int nMaxPopulation = nPopulation;
	printf("LoadData - GPU\n");
	CleanCudaMemory();
	PrepareCudaMemory();
	MoveCityLocToCudaMemory();	
	
}

void CGPUTSPSolver::initSolver(void) {

	GeneInitCudaMemory();

	nGeneration = 0;
	Temperature = 100.0;
	crossoverMethod = BCSCX;
	recordBroken = false;
	bHeating = false;
	
	for (int i = 0; i < nPopulation; i++) {
		h_fitness[i] = 0;
	}
	printf("gene data at cuda memory generated\n");
	
	
}

void CGPUTSPSolver::computeFitnessOf(int idx) {
	

	int blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	d_computeFitnessOf(blocksPerGrid, THREADSPERBLOCK, idx, d_CityLoc, d_gene, nPopulation, nCities, d_fitness, d_distanceSeq);

	//safeCuda(cudaGetLastError(), "Get Last Error at ComputeFitnessOf");	
	
	
	char msg[256];
	sprintf(msg, "device to host memory copy: fitOf(%d) d_distanceSeq -> h_distance", idx);
	safeCuda(cudaMemcpy(h_distanceSeq, d_distanceSeq, sizeof(int)*nCities, cudaMemcpyDeviceToHost), msg);
	
	mFitness[idx] = 0;
	for (int i = 0; i < nCities; i++) mFitness[idx] += h_distanceSeq[i];

	sprintf(msg, "host to device memory copy mFitness -> d_fitness");
	safeCuda(cudaMemcpy(d_fitness, mFitness, sizeof(int)*nPopulation, cudaMemcpyHostToDevice), msg);
	//printf("fit(%d) = %d", idx, mFitness[idx]);
}



void CGPUTSPSolver::computeFitness(void) {

	
	recordBroken = false;	

	int blocksPerGrid = (nPopulation + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	d_computeFitnessAll(blocksPerGrid, THREADSPERBLOCK, d_CityLoc, d_gene, nPopulation, nCities, d_fitness);

	char msg[256];
	sprintf(msg, "device to host memory copy (%d genes): d_fitness -> mFitness", nPopulation);
	safeCuda(cudaMemcpy(mFitness, d_fitness, sizeof(int)*nPopulation, cudaMemcpyDeviceToHost), msg);
	

	int nGroups = nNumberOfGroups;
	int nMemberOfAGroup = nPopulation / nGroups;

	blocksPerGrid;

	for (int g = 0; g < nGroups; g++) {

		int start = g*nMemberOfAGroup;
		int end = (g == nGroups - 1) ? nPopulation - 1 : start + nMemberOfAGroup;
		int nMemberOfAGroup = end - start;

		bestFitness = mFitness[start];
		bestGeneIdx = start;

		for (int i = start; i < end; i++) {
			if (mFitness[i] < bestFitness) { bestGeneIdx = i; bestFitness = mFitness[i]; }
		}
		blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		d_copyGene(blocksPerGrid, THREADSPERBLOCK, start, bestGeneIdx, d_gene, nCities);
	}

	bestFitness = mFitness[0];
	bestGeneIdx = 0;
	for (int i = nMemberOfAGroup; i < nPopulation; i += nMemberOfAGroup) {
		if (mFitness[i] < bestFitness) { bestGeneIdx = i; bestFitness = mFitness[i]; }
	}

	safeCuda(cudaMemcpy(currentBestGene, d_gene + (bestGeneIdx*nCities), sizeof(int)*nCities, cudaMemcpyDeviceToHost), "get the best gene in the generation");

	if (nGeneration <= 1) {
		fitRecord = mFitness[bestGeneIdx];
		safeCuda(cudaMemcpy(recordHolder, d_gene+(bestGeneIdx*nCities), sizeof(int)*nCities, cudaMemcpyDeviceToHost), "record holder update 1");
	}
	else {
		if (mFitness[bestGeneIdx]<fitRecord) {
			fitRecord = mFitness[bestGeneIdx];
			safeCuda(cudaMemcpy(recordHolder, d_gene+(bestGeneIdx*nCities), sizeof(int)*nCities, cudaMemcpyDeviceToHost), "record holder update 2");
			recordBroken = true;
		}
		else {
			recordBroken = false;
		}
	}


}


void CGPUTSPSolver::nextGeneration(void) {
	nGeneration++;
	int nGroups = nNumberOfGroups;
	int nMemberOfAGroup = nPopulation / nGroups;



	
	/*if ((nGeneration / 10) % 2 == 0) {
		for (int g = 0; g < nGroups; g++) {
			int start = g*nMemberOfAGroup;
			fixGene(start);
		}
		return;
	}*/
	

	// linear cooling
	if (bHeating) {
		float cooling = 100.0 / nCycleGeneration;
		float prevTemp = Temperature;
		Temperature -= cooling;
		if (Temperature < 0) Temperature = 100.0;
	}
	else { Temperature = 0.0; }




	int blocksPerGrid;

	for (int g = 0; g < nGroups; g++) {

		int start = g*nMemberOfAGroup;
		int end = (g == nGroups - 1) ? nPopulation - 1 : start + nMemberOfAGroup;
		int nMemberOfAGroup = end - start;
		// 1. gene competition
		blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		for (int i = 0; i < nMemberOfAGroup/2; i++) {
			int winner = start + i*2;
			int loser  = winner + 1;
			if (mFitness[winner]> (1.0 - Temperature / 1000.0)*mFitness[loser]) { winner = loser; }
			//if (bHeating && rangeRandomf(0.0, 100.0) < Temperature / 5.0) winner = loser;
			// elements of a gene sequence are parallelly moved from the location "winner" to "i" in the pool
			d_copyGene(blocksPerGrid, THREADSPERBLOCK, start + i, winner, d_gene, nCities);
		}
		if (nMemberOfAGroup % 2)
			d_copyGene(blocksPerGrid, THREADSPERBLOCK, start + nMemberOfAGroup / 2, start + nMemberOfAGroup - 1, d_gene, nCities);

	}
		
	// 2. apply crossover
	

	// 2.1 initialize aux memories for crossover (parallel processing of elements in a single gene)
	blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	for (int i = 0; i < nPopulation; i++) {
		// N(nCities) elements are parallelly processed
		d_initAuxMem(blocksPerGrid, THREADSPERBLOCK, nCities, i, d_gene, d_orderOfCity, d_fJump, d_bJump);
	}


	// 2.2 crossover
	int nCrossover = nMemberOfAGroup / 4;
	dim3 threads(nCrossover , nGroups);
	blocksPerGrid = (threads.x * threads.y + THREADSPERBLOCK - 1 ) / THREADSPERBLOCK;

	if (this->crossoverMethod == BCSCX) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossover(blocksPerGrid, threads, i, nPopulation, nGroups, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}
	else if (this->crossoverMethod == ABCSCX) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossoverABCSCX(blocksPerGrid, threads, i, 0, nPopulation, nGroups, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
		blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		for (int i = 0; i < nPopulation; i++) {
			// N(nCities) elements are parallelly processed
			d_initAuxMem(blocksPerGrid, THREADSPERBLOCK, nCities, i, d_gene, d_orderOfCity, d_fJump, d_bJump);
		}
		blocksPerGrid = (threads.x * threads.y + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossoverABCSCX(blocksPerGrid, threads, i, 1, nPopulation, nGroups, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}
	else if (this->crossoverMethod == MIXED) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossover(blocksPerGrid, threads, i, nPopulation, nGroups, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
		blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		for (int i = 0; i < nPopulation; i++) {
			// N(nCities) elements are parallelly processed
			d_initAuxMem(blocksPerGrid, THREADSPERBLOCK, nCities, i, d_gene, d_orderOfCity, d_fJump, d_bJump);
		}
		blocksPerGrid = (threads.x * threads.y + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossoverABCSCX(blocksPerGrid, threads, i, 1, nPopulation, nGroups, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}
	

	for (int g = 0; g < nGroups; g++) {
		int start = g*nMemberOfAGroup;		
		fixGene(start);
	}



	// 3. mutate gene
	
	for (int g = 0; g < nGroups; g++) {

		int start = g*nMemberOfAGroup;
		int end = (g == nGroups - 1) ? nPopulation - 1 : start + nMemberOfAGroup;
		int nMemberOfAGroup = end - start;

		blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		int mutationPercentage = 10;
		for (int i = start+1; i < end; i++) {
			if (rangeRandomi(0, 100) < mutationPercentage) {
				int idxA = rangeRandomi(1, nCities - 1);
				int idxB = idxA;
				while (idxB == idxA) idxB = rangeRandomi(1, nCities - 1);
				if (idxA > idxB) { int t = idxA; idxA = idxB; idxB = t; }
				blocksPerGrid = ((idxB - idxA + 1) / 2 + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
				d_reverseSubGene(blocksPerGrid, THREADSPERBLOCK, i, idxA, idxB, d_gene, nCities);
			}
		}
	}
	
	
		
	
	/*
	if (nGeneration % 20) {
		for (int g = 0; g < nGroups; g++) intergroupMarriage(g);
			
	}
	*/

	/*
	static int count = 0;
	int maxLocalMin = 128;
	int nCycle = 100;
	if (nGeneration>0 && nGeneration % nCycle == 0) {
		this->copySolution(localMinima[count++]);
		nLocalMin = count;
		geneReset();
		if (count == maxLocalMin) {
			char fname[256];
			sprintf(fname, "localMinima%d.txt", nCities);
			FILE *f = fopen(fname, "w");
			fprintf(f, "%d\n%d\n", maxLocalMin, nCities);
			for (int i = 0; i < maxLocalMin; i++) {
				for (int j = 0; j < nCities; j++) {
					fprintf(f, "%d\n", localMinima[i][j] + 1);
				}
			}
			nLocalMin = 0; count = 0;
			exit(0);
		}
	}
	*/
	
	
	

	
}


void CGPUTSPSolver::intergroupMarriage(int groupIdx) {

	int blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	int nMemberOfAGroup = nPopulation / nNumberOfGroups;
	int fromIdx = groupIdx * nMemberOfAGroup;
	int toIdx = ((groupIdx + 1) % nNumberOfGroups) * nMemberOfAGroup;
	d_copyGene(blocksPerGrid, THREADSPERBLOCK, toIdx + 1, /* nMemberOfAGroup-1,*/ fromIdx, d_gene, nCities);
}

/*
void CGPUTSPSolver::nextGeneration(void) {
	nGeneration++;

	// linear cooling
	if (bHeating) {
		float cooling = 100.0 / nCycleGeneration;
		float prevTemp = Temperature;
		Temperature -= cooling;
		if (Temperature < 0) Temperature = 100.0;
	}
	else { Temperature = 0.0; }



	// 1. gene competition
	int blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	for (int i = 0; i < nPopulation/2; i ++) {
		int winner = i*2;
		int loser = winner + 1;
		if (mFitness[winner]>(1.0-Temperature/1000.0)*mFitness[loser]) { winner = i*2 + 1; loser = i*2; }
		//if (bHeating && rangeRandomf(0.0, 100.0) < Temperature / 5.0) winner = loser;
		// elements of a gene sequence are parallelly moved from the location "winner" to "i" in the pool
		d_copyGene(THREADSPERBLOCK, blocksPerGrid, i, winner, d_gene, nCities);
	}
	if (nPopulation%2) 
		d_copyGene(THREADSPERBLOCK, blocksPerGrid, nPopulation / 2, nPopulation - 1, d_gene, nCities);


	// 2. apply crossover
	int nCrossover = nPopulation / 4;

	// 2.1 initialize aux memories for crossover (parallel processing of elements in a single gene)
	blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	for (int i = 0; i < nPopulation; i++) {
		// N(nCities) elements are parallelly processed
		d_initAuxMem(THREADSPERBLOCK, blocksPerGrid, nCities, i, d_gene, d_orderOfCity, d_fJump, d_bJump);
	}

	// 2.2 crossover
	blocksPerGrid = ( nCrossover + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	if (this->crossoverMethod == CROSSOVERMETHOD::BCSCX) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossover(THREADSPERBLOCK, blocksPerGrid, i, nCrossover, 1,  nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}
	else if (this->crossoverMethod == CROSSOVERMETHOD::ABCSCX) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossoverABCSCX(THREADSPERBLOCK, blocksPerGrid, i, nCrossover, 1, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}
	else if (this->crossoverMethod == CROSSOVERMETHOD::MIXED) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossover(THREADSPERBLOCK, blocksPerGrid, i, nCrossover, 1, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}

	// 2.3 initialize aux memories for crossover (parallel processing of elements in a single gene)
	blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	for (int i = 0; i < nPopulation; i++) {
		// N(nCities) elements are parallelly processed
		d_initAuxMem(THREADSPERBLOCK, blocksPerGrid, nCities, i, d_gene, d_orderOfCity, d_fJump, d_bJump);
	}
	// 2.4 crossover
	blocksPerGrid = (nCrossover + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	if (this->crossoverMethod == CROSSOVERMETHOD::BCSCX) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossover(THREADSPERBLOCK, blocksPerGrid, i, nCrossover, 2, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}
	else if (this->crossoverMethod == CROSSOVERMETHOD::ABCSCX) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossoverABCSCX(THREADSPERBLOCK, blocksPerGrid, i, nCrossover, 2, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}
	else if (this->crossoverMethod == CROSSOVERMETHOD::MIXED) {
		for (int i = 1; i < nCities; i++) { // should start from 1 !!! ( gene[i*nCities+0] always starts with 0 )
			// create i-th elements of offsprings ( parallel processing of all genes for constructing one more offspring gene element)
			d_crossoverABCSCX(THREADSPERBLOCK, blocksPerGrid, i, nCrossover, 2, nCities, d_gene, d_CityLoc, d_orderOfCity, d_fJump, d_bJump);
		}
	}

	// 3. mutate gene
	blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	int mutationPercentage = 35;
	for (int i = nPopulation * 3 / 4; i < nPopulation; i++) {
		if (rangeRandomi(0, 100) < mutationPercentage) {
			int idxA = rangeRandomi(1, nCities - 1);
			int idxB = idxA;
			while (idxB == idxA) idxB = rangeRandomi(1, nCities - 1);
			if (idxA > idxB) { int t = idxA; idxA = idxB; idxB = t; }
			blocksPerGrid = ((idxB - idxA + 1) / 2 + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
			d_reverseSubGene(THREADSPERBLOCK, blocksPerGrid, i, idxA, idxB, d_gene, nCities);
		}
	}
	fixGene(0);
}
*/

void CGPUTSPSolver::fixGene(int idx) {
	

	int idxA;
	int idxB;
	int search = 0;
	int maxSearch = (int)sqrt(nCities);


	int distOrg, distNew;
	int maxGain;
	int iForMaxGain = -1, jForMaxGain = -1;
	int iCity, jCity;
	bool bIntersectFound;
	bool bMovableFound;


	// Start: city move -----------------------------------
	maxGain = 0;
	bMovableFound = false;
	idxA = rangeRandomi(1, nCities - 2 - maxSearch);
	idxB = rangeRandomi(1, nCities - 2 - maxSearch);
	int *geneFrag1 = (int *) malloc(sizeof(int)*(maxSearch + 2));
	int *geneFrag2 = (int *) malloc(sizeof(int)*(maxSearch + 2));
	safeCuda(cudaMemcpy(geneFrag1, d_gene + (idx*nCities) + idxA - 1, sizeof(int)*(maxSearch + 2), cudaMemcpyDeviceToHost), "get the gene fragment 1");
	safeCuda(cudaMemcpy(geneFrag2, d_gene + (idx*nCities) + idxB - 1, sizeof(int)*(maxSearch + 2), cudaMemcpyDeviceToHost), "get the gene fragment 2");
	int v1, v2;
	int Prev1, Next1, Next2;
	
	for (int i = 0; i<maxSearch; i++) {
		for (int j = 0; j<maxSearch; j++) {

			Prev1 = geneFrag1[i];
			v1 =    geneFrag1[i + 1];
			Next1 = geneFrag1[i + 2];
			v2 =    geneFrag2[j + 1];
			Next2 = geneFrag2[j + 2];
			
			
			distOrg = cityLocData->cityDistance(Prev1, v1) + cityLocData->cityDistance(v1, Next1) + cityLocData->cityDistance(v2, Next2);
			distNew = cityLocData->cityDistance(Prev1, Next1) + cityLocData->cityDistance(v2, v1) + cityLocData->cityDistance(v1, Next2);

			int gain = distOrg - distNew;
			if (gain>maxGain) {
				bMovableFound = true;
				iForMaxGain = idxA + i;
				iCity = v1;
				jForMaxGain = idxB + j;
				jCity = v2;
				maxGain = gain;
			}

		}
	}

	free(geneFrag1);
	free(geneFrag2);

	if (bMovableFound) {
		if (iForMaxGain > jForMaxGain) { 
			int t = iForMaxGain; iForMaxGain = jForMaxGain; jForMaxGain = t; 
			t = iCity; iCity = jCity; jCity = t;
		}
		int blocksPerGrid = (nCities + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		d_createACityShiftedGene(THREADSPERBLOCK, blocksPerGrid, nCities, iCity, iForMaxGain, jForMaxGain, d_gene, idx, d_aGene);
		d_copyBack(THREADSPERBLOCK, blocksPerGrid, nCities, d_gene, idx, d_aGene);		
	}
	// End: city move -----------------------------------


	// 2-opt
	maxGain = 0;
	bIntersectFound = false;
	geneFrag1 = (int *)malloc(sizeof(int)*(maxSearch + 1));
	geneFrag2 = (int *)malloc(sizeof(int)*(maxSearch + 1));
	safeCuda(cudaMemcpy(geneFrag1, d_gene + (idx*nCities) + idxA, sizeof(int)*(maxSearch + 1), cudaMemcpyDeviceToHost), "get the gene fragment 3");
	safeCuda(cudaMemcpy(geneFrag2, d_gene + (idx*nCities) + idxB, sizeof(int)*(maxSearch + 1), cudaMemcpyDeviceToHost), "get the gene fragment 4");
	int v1n, v2n;
	for (int i = 0; i<maxSearch; i++) {
		for (int j = 0; j<maxSearch; j++) {
			v1  = geneFrag1[i];
			v1n = geneFrag1[i + 1];
			v2  = geneFrag2[j];
			v2n = geneFrag2[j + 1];
			if (v1 == v2) continue;
			distOrg = cityLocData->cityDistance(v1, v1n) + cityLocData->cityDistance(v2, v2n);
			distNew = cityLocData->cityDistance(v1, v2) + cityLocData->cityDistance(v1n, v2n);
			int gain = distOrg - distNew;
			if (gain>maxGain) {
				bIntersectFound = true;
				iForMaxGain = idxA+i;
				jForMaxGain = idxB+j;
				maxGain = gain;
			}
		}
	}

	if (bIntersectFound) {
		if (iForMaxGain > jForMaxGain) { int t = iForMaxGain; iForMaxGain = jForMaxGain; jForMaxGain = t; }
		int blocksPerGrid = ((jForMaxGain - iForMaxGain) / 2 + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		d_reverseSubGene(THREADSPERBLOCK, blocksPerGrid, idx, iForMaxGain+1, jForMaxGain, d_gene, nCities);
	}

	free(geneFrag1);
	free(geneFrag2);

	maxGain = 0;
	bIntersectFound = false;

	geneFrag1 = (int *)malloc(sizeof(int)*nCities);
	safeCuda(cudaMemcpy(geneFrag1, d_gene + (idx*nCities), sizeof(int)*nCities, cudaMemcpyDeviceToHost), "get the gene fragment 5");

	for (int i = idxA; i< idxA+maxSearch; i++) {
		for (int j = 0; j<maxSearch; j++) {
			v1  = geneFrag1[i];
			v1n = geneFrag1[i + 1];
			v2  = geneFrag1[(i + j + 2) % nCities];
			v2n = geneFrag1[(i + j + 3) % nCities];

			int gain;
			if (v2 != 0) {
				distOrg = cityLocData->cityDistance(v1, v1n) + cityLocData->cityDistance(v2, v2n);
				distNew = cityLocData->cityDistance(v1, v2) + cityLocData->cityDistance(v1n, v2n);
				gain = distOrg - distNew;
				if (gain>maxGain) {
					bIntersectFound = true;
					maxGain = gain;
					iForMaxGain = i;
					jForMaxGain = (i + j + 2) % nCities;
				}
			}

			int sIdx = i - 2 - j;
			if (sIdx<0) sIdx += nCities;
			v2  = geneFrag1[sIdx];
			v2n = geneFrag1[(sIdx + 1) % nCities];

			if (v2 != 0) {
				distOrg = cityLocData->cityDistance(v1, v1n) + cityLocData->cityDistance(v2, v2n);
				distNew = cityLocData->cityDistance(v1, v2) + cityLocData->cityDistance(v1n, v2n);
				gain = distOrg - distNew;
				if (gain>maxGain) {
					bIntersectFound = true;
					maxGain = gain;
					iForMaxGain = i;
					jForMaxGain = sIdx;
				}
			}
		}
	}

	if (bIntersectFound) {
		if (iForMaxGain > jForMaxGain) { int t = iForMaxGain; iForMaxGain = jForMaxGain; jForMaxGain = t; }
		int blocksPerGrid = ((jForMaxGain - iForMaxGain) / 2 + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		d_reverseSubGene(THREADSPERBLOCK, blocksPerGrid, idx, iForMaxGain + 1, jForMaxGain, d_gene, nCities);
	}

	free(geneFrag1);

	computeFitnessOf(idx);
}



void CGPUTSPSolver::copySolution(int *SolutionCpy) {
	if (!SolutionCpy) return;

	safeCuda(cudaMemcpy(SolutionCpy, d_gene + (bestGeneIdx*nCities), sizeof(int)*nCities, cudaMemcpyDeviceToHost), "copy best gene"); 
	
}

void CGPUTSPSolver::LoadSolution(const char *fname) {
	int cityId;
	FILE *fInput = fopen(fname, "r");
	if (fInput == NULL) {
		printf("file not found : %s\n", fname);
		char pathStr[256];
		GetCurrentDir(pathStr, sizeof(pathStr));
		printf("working dir: %s\n", pathStr);

		exit(1);
	}

	printf("file loading started...\n");

	int N;

	fscanf(fInput, "%d\n", &N);
	for (int i = 0; i<N; i++) {
		fscanf(fInput, "%d", &cityId);
		gene[0][i] = cityId - 1;
	}
	
	safeCuda(cudaMemcpy(d_gene, gene[0], sizeof(int)*nCities, cudaMemcpyHostToDevice), "loaded solution to device gene");
}


void CGPUTSPSolver::LoadLocalMinima(const char *fname) {
	int cityId;
	FILE *fInput = fopen(fname, "r");
	if (fInput == NULL) {
		printf("file not found : %s\n", fname);
		char pathStr[256];
		GetCurrentDir(pathStr, sizeof(pathStr));
		printf("working dir: %s\n", pathStr);

		exit(1);
	}

	printf("file loading started...\n");

	int nGenes;
	int nCities;

	fscanf(fInput, "%d\n", &nGenes);
	fscanf(fInput, "%d\n", &nCities);

	int nMemberOfAGroup = nPopulation / nNumberOfGroups;
	
	
	for (int i = 0; i < nGenes && i < nMemberOfAGroup ; i++) {
		for (int j = 0; j < nCities; j++) {
			fscanf(fInput, "%d", &cityId);
			gene[0][j] = cityId - 1;
		}
		for (int g = 0; g < nNumberOfGroups; g++) {
			int start = g*nMemberOfAGroup;
			safeCuda(cudaMemcpy(d_gene + (start + i)*nCities, gene[0], sizeof(int)*nCities, cudaMemcpyHostToDevice), "load local minima to device gene");
		}
	}
	

	
}

void CGPUTSPSolver::geneReset(void) {
	GeneInitCudaMemory();
}