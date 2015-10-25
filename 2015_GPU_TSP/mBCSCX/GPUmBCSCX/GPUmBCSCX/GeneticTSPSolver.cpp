
//
//  GeneticTSPSolver.cpp
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#include "GeneticTSPSolver.h"
#include <math.h>

CCitySearchIndex::CCitySearchIndex(int numberOfCities) {
    nSize = numberOfCities;
    validity = new bool[numberOfCities];
    forwardJump = new int[numberOfCities];
    backwardJump = new int[numberOfCities];
    for(int i=0;i<numberOfCities;i++) {
        validity[i] = true;
        forwardJump[i] = 1;
        backwardJump[i] = 1;
    }
}
CCitySearchIndex::~CCitySearchIndex() {
    delete[] validity;
    delete[] forwardJump;
    delete[] backwardJump;
}

void CCitySearchIndex::invalidate(int i) {
    validity[i] = false;
    int updateFowardIdx = i - backwardJump[i];
    if(updateFowardIdx < 0 ) updateFowardIdx += nSize;
    forwardJump[updateFowardIdx] += forwardJump[i];
    int updateBackwardIdx = (i + forwardJump[i])%nSize;
    backwardJump[updateBackwardIdx] += backwardJump[i];
}

int CCitySearchIndex::getForwardJump(int i) {
    return forwardJump[i];
}

int CCitySearchIndex::getBackwardJump(int i) {
    return backwardJump[i];
}

CGeneticTSPSolver::CGeneticTSPSolver() {
	cityLocData = NULL;
	nPopulation = 0;
	nNumberOfGroups = 1;

	nCities = 0;
	mFitness = NULL;


	gene = NULL;
}

CGeneticTSPSolver::CGeneticTSPSolver(CCityLocData *inputData, int nGenes, int nGroups) {
    cityLocData = inputData;
    nPopulation = nGenes;
    nNumberOfGroups = nGroups;
    
	nCities = cityLocData->numCities;
	mFitness = new int[nPopulation];
    

    gene = new int*[nPopulation];
    
    initSolver();
    
}

CGeneticTSPSolver::~CGeneticTSPSolver() {
	if (cityLocData) cityLocData = NULL;
	if (mFitness) delete[] mFitness;
	if (gene) {
		for (int i = 0; i < nPopulation; i++) {
			delete[] gene[i];
		}
		delete[] gene;
	}
}

void CGeneticTSPSolver::RemoveData(void) {
	if (cityLocData) cityLocData = NULL;
	if (mFitness) delete[] mFitness;
	if (gene) {
		for (int i = 0; i < nPopulation; i++) {
			delete[] gene[i];
		}
		delete[] gene;
	}
	nPopulation = 0;
	nNumberOfGroups = 1;
}

void CGeneticTSPSolver::LoadData(CCityLocData *inputData, int nGenes, int nGroups) {
    
    // safe cleaning
    if(gene) {
        for(int i=0;i<nPopulation;i++) {
            delete[] gene[i];
        }
        delete[] gene;
    }
    if(mFitness) delete[] mFitness;
    
    //////////////////
    
	cityLocData = inputData;
	nPopulation = nGenes;
	nNumberOfGroups = nGroups;

	nCities = cityLocData->numCities;
    
    mFitness = new int[nPopulation];

    
	gene = new int*[nPopulation];

	initSolver();

}

void CGeneticTSPSolver::LoadSolution(char *fname) {
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
        gene[0][i] = cityId-1;
    }
    computeFitnessOf(0);
    
    printf("solution loaded\n");
}


void CGeneticTSPSolver::initSolver(void) {
    nGeneration = 0;
    
    for (int i=0; i<nPopulation; i++) {
        gene[i] = new int[nCities];
    }
    for (int i=0; i<nPopulation; i++) {
        for (int j=0; j<nCities  ; j++) gene[i][j] = j;
    }
    for (int i=0; i<nPopulation; i++) {
        shuffleGene(i, nCities/2);
    }
}

//void CGeneticTSPSolver::reverseGene(int idx) {
//    int max=nCities/2;
//    for(int i=1;i<=max;i++) {
//        swap(gene[idx][i], gene[idx][nCities-i]);
//    }
//}

void CGeneticTSPSolver::shuffleGene(int idx, int nShuffle) {
    int idxA, idxB;
    for(int i=1;i<nShuffle;i++) {
        idxA = rangeRandomi(1, nCities-1);
        idxB = rangeRandomi(1, nCities-1);
        
        swap(gene[idx][idxA], gene[idx][idxB]);
   }
}

void CGeneticTSPSolver::swapGene(int idxA, int idxB) {
    for (int i=0; i<nCities; i++) swap(gene[idxA][i], gene[idxB][i]);
    
    swap(mFitness[idxA], mFitness[idxB]);
}

void CGeneticTSPSolver::copyGene(int idxA, int idxB) {
    for (int i=0; i<nCities; i++) {
        gene[idxB][i]=gene[idxA][i];
    }
    mFitness[idxB] = mFitness[idxA];
}

int CGeneticTSPSolver::getLegitimateNodeBCSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx) {
    int idx = orderOfCity[curCity];
    int forward = (idx + citySearchIdx.getForwardJump(idx))%nCities;
    int backward = idx - citySearchIdx.getBackwardJump(idx);
    while(backward<0) backward+=nCities;
    
    int nCity1 = cityTour[forward];
    int nCity2 = cityTour[backward];
	int nextCity = (cityLocData->cityDistance(curCity, nCity1) < cityLocData->cityDistance(curCity, nCity2)) ? nCity1 : nCity2;
    return nextCity;
    

}

void CGeneticTSPSolver::mutate(int parent, int idx) {
    
    copyGene(parent, idx);
    
    // inverting a section
    
    
    int idxA;
    int idxB;
    idxA = rangeRandomi(0, nCities-1);
    idxB = rangeRandomi(0, nCities-1);
    if(idxA>idxB) { int T = idxA; idxA = idxB; idxB = T; }
    int half = (idxB-1-idxA)/2;
    for(int i=0;i<half;i++) {
        swap(gene[idx][idxA+1+i], gene[idx][idxB-1-i]);
    }
    
    computeFitnessOf(idx);
}


void CGeneticTSPSolver::fixGene(int idx) {
    
    // inverting a section
    
    
    int idxA;
    int idxB;
    int search = 0;
    int maxSearch = (int) (sqrt(nCities)+0.5);
    maxSearch*=10;
    idxA = rangeRandomi(0, nCities-1-maxSearch);
    idxB = rangeRandomi(0, nCities-1-maxSearch);

    int distOrg, distNew;
    int maxGain = 0;
    int iForMaxGain=-1, jForMaxGain=-1;
    bool bIntersectFound = false;
    for(int i=0;i<maxSearch;i++) {
        for(int j=i+1;j<maxSearch;j++) {
            int v1 = idxA+i;
            int v2 = idxB+j;
            distOrg = cityLocData->cityDistance(gene[idx][v1], gene[idx][v1+1]) + cityLocData->cityDistance(gene[idx][v2], gene[idx][v2+1]);
            distNew = cityLocData->cityDistance(gene[idx][v1], gene[idx][v2])   + cityLocData->cityDistance(gene[idx][v1+1], gene[idx][v2+1]);
            int gain = distOrg - distNew;
            if(gain>maxGain) {
                bIntersectFound=true;
                iForMaxGain = v1;
                jForMaxGain = v2;
                maxGain = gain;
            }
            
        }
    }
    for(int i=0;i<nCities-1;i++) {
        for(int j=i+1;j<maxSearch && j<nCities-1;j++) {
            int v1 = i;
            int v2 = j;
            distOrg = cityLocData->cityDistance(gene[idx][v1], gene[idx][v1+1]) + cityLocData->cityDistance(gene[idx][v2], gene[idx][v2+1]);
            distNew = cityLocData->cityDistance(gene[idx][v1], gene[idx][v2])   + cityLocData->cityDistance(gene[idx][v1+1], gene[idx][v2+1]);
            int gain = distOrg - distNew;
            if(gain>maxGain) {
                bIntersectFound=true;
                iForMaxGain = v1;
                jForMaxGain = v2;
                maxGain = gain;
            }
            
        }
    }
    
    if(bIntersectFound) {
        if(iForMaxGain > jForMaxGain) { int t = iForMaxGain; iForMaxGain=jForMaxGain; jForMaxGain=t; }
        int half = (jForMaxGain-iForMaxGain)/2;
        for(int i=0;i<half;i++) {
            swap(gene[idx][iForMaxGain+1+i], gene[idx][jForMaxGain-i]);
        }
        
    }
    computeFitnessOf(idx);
}

/*
void CGeneticTSPSolver::fixGene(int idx) {
    
    // inverting a section
    
    
    int idxA;
    int idxB;
    int search = 0;
    int distOrg, distNew;
    bool bIntersectFound = false;
    while(search < nCities && !bIntersectFound) {
        idxA = rangeRandomi(0, nCities-2);
        idxB = rangeRandomi(0, nCities-2);
        if(idxA>idxB) { int T = idxA; idxA = idxB; idxB = T; }
        if(idxA==idxB) continue;
        
        distOrg = cityLocData->cityDistance(gene[idx][idxA], gene[idx][idxA+1]) + cityLocData->cityDistance(gene[idx][idxB], gene[idx][idxB+1]);
        distNew = cityLocData->cityDistance(gene[idx][idxA], gene[idx][idxB])   + cityLocData->cityDistance(gene[idx][idxA+1], gene[idx][idxB+1]);
        if(distNew<distOrg) {
            bIntersectFound=true;
        }
        search++;
    }
    
    
    if(bIntersectFound) {
        int half = (idxB-idxA)/2;
        for(int i=0;i<half;i++) {
            swap(gene[idx][idxA+1+i], gene[idx][idxB-i]);
        }
        
    }
    computeFitnessOf(idx);
}
 */


void CGeneticTSPSolver::crossoverBCSCX(int idxA, int idxB, int idxC) {


    int candidate1, candidate2;
	int curCity = 0;
	int nextCity;
	bool *cityValid = new bool[nCities];
	int  *crossover = new int[nCities];
	crossover[0] = 0;

    
	for(int i=1;i<nCities;i++) {
		cityValid[i] = true;
	}	cityValid[0] = false;

    CCitySearchIndex citySearchIdx_A(nCities);
    CCitySearchIndex citySearchIdx_B(nCities);
    citySearchIdx_A.invalidate(0);
    citySearchIdx_B.invalidate(0);
    
    int *orderOfCity_A = new int[nCities];
    int *orderOfCity_B = new int[nCities];
    
    for (int i=0; i<nCities; i++) {
        int city;
        city = gene[idxA][i]; orderOfCity_A[city] = i;
        city = gene[idxB][i]; orderOfCity_B[city] = i;
    }

    for(int i=1; i<nCities; i++) {
		candidate1 = getLegitimateNodeBCSCX(curCity, gene[idxA], orderOfCity_A, citySearchIdx_A);
		candidate2 = getLegitimateNodeBCSCX(curCity, gene[idxB], orderOfCity_B, citySearchIdx_B);
		if (cityLocData->cityDistance(curCity, candidate1) < cityLocData->cityDistance(curCity, candidate2))
			nextCity = candidate1;
		else nextCity = candidate2;
		crossover[i]=nextCity;
		cityValid[nextCity] = false;
		curCity = nextCity;
        citySearchIdx_A.invalidate(orderOfCity_A[nextCity]);
        citySearchIdx_B.invalidate(orderOfCity_B[nextCity]);
	}

    for(int i=0;i<nCities;i++) {
        gene[idxC][i]=crossover[i];
    }
	delete[] cityValid;
	delete[] crossover;
    
    delete[] orderOfCity_A;
    delete[] orderOfCity_B;
    
    computeFitnessOf(idxC);
}

void CGeneticTSPSolver::crossoverABCSCX(int idxA, int idxB, int idxC) {
    
 
    bool bUse1stGene = true;
    
    int curCity = 0;
    int nextCity;
    bool *cityValid = new bool[nCities];
    int  *crossover = new int[nCities];
    crossover[0] = 0;
    
    
    for(int i=1;i<nCities;i++) {
        cityValid[i] = true;
    }	cityValid[0] = false;
    
    CCitySearchIndex citySearchIdx_A(nCities);
    CCitySearchIndex citySearchIdx_B(nCities);
    citySearchIdx_A.invalidate(0);
    citySearchIdx_B.invalidate(0);
    
    int *orderOfCity_A = new int[nCities];
    int *orderOfCity_B = new int[nCities];
    
    for (int i=0; i<nCities; i++) {
        int city;
        city = gene[idxA][i]; orderOfCity_A[city] = i;
        city = gene[idxB][i]; orderOfCity_B[city] = i;
    }
    
    for(int i=1; i<nCities; i++) {
        if(bUse1stGene ) {
            nextCity = getLegitimateNodeBCSCX(curCity, gene[idxA], orderOfCity_A, citySearchIdx_A);
        }
        else {
            nextCity = getLegitimateNodeBCSCX(curCity, gene[idxB], orderOfCity_B, citySearchIdx_B);
        }
        crossover[i]=nextCity;
        cityValid[nextCity] = false;
        curCity = nextCity;
        citySearchIdx_A.invalidate(orderOfCity_A[nextCity]);
        citySearchIdx_B.invalidate(orderOfCity_B[nextCity]);
        bUse1stGene = bUse1stGene?false:true;
    }
    
    for(int i=0;i<nCities;i++) {
        gene[idxC][i]=crossover[i];
    }
    delete[] cityValid;
    delete[] crossover;
    
    delete[] orderOfCity_A;
    delete[] orderOfCity_B;
    
    
    
    computeFitnessOf(idxC);
}




void CGeneticTSPSolver::copySolution(int *SolutionCpy) {
    if(!SolutionCpy) return;
    
    for (int i=0; i<nCities; i++) {
        SolutionCpy[i] = gene[bestGeneIdx][i];
    }
}


void CGeneticTSPSolver::computeFitnessOf(int idx) {
    
    mFitness[idx] = 0;
    for(int j=0;j<nCities;j++) {
        mFitness[idx] += cityLocData->cityDistance( gene[idx][j] , gene[idx][(j+1)%nCities] );
    }
    
}

void CGeneticTSPSolver::computeFitness(void) {
    
    int nMemberOfAGroup = nPopulation/nNumberOfGroups;
   
   
    for(int g=0;g<nNumberOfGroups;g++) { // for every local group
        
        int start = g*nMemberOfAGroup;
        int end = (g==nNumberOfGroups-1)?nPopulation: start+nMemberOfAGroup;
        
        int localBestFitness = mFitness[start];
        int   localBestGeneIdx = start;
        
        for(int i=start;i<end;i++) {
            computeFitnessOf(i);
            if(mFitness[i]<localBestFitness) { localBestFitness = mFitness[i]; localBestGeneIdx = i; }
            debegMessage("fit (%d) = %d\n", i, mFitness[i]);
        }
        
        // move the best in the local group to be the first gene in the group
        debegMessage("group(%d) - swapping %d (%d) and %d (%d)\n", g, start, mFitness[start], localBestGeneIdx, localBestFitness);
        swapGene(start, localBestGeneIdx);
        
    }
    
    

    
    bestFitness = mFitness[0];
    bestGeneIdx = 0;

    
    // find the best in the total population
    for(int i=0;i<nPopulation;i+=nMemberOfAGroup) {
        if(mFitness[i]<bestFitness) { bestGeneIdx = i; bestFitness = mFitness[i]; }
    }
    
    fixGene(bestGeneIdx);
    computeFitnessOf(bestGeneIdx);
    debegMessage("bestGene = %d (%d)\n", bestGeneIdx, bestFitness);
    
}

void CGeneticTSPSolver::intergroupMarriage(void) {
    
    int nMemberOfAGroup = nPopulation/nNumberOfGroups;
    // best gene migration
    if(nNumberOfGroups>1) {
        for(int i=0;i<nNumberOfGroups;i++) {
            int idxA = i;
            int idxB = (i + rangeRandomi(1, nNumberOfGroups-1))%nNumberOfGroups;
            
            idxA *= nMemberOfAGroup;
            idxB *= nMemberOfAGroup;
            
            // best gene inter-group marrage
            crossoverBCSCX(idxA, idxB, idxA+nMemberOfAGroup-1);
            crossoverABCSCX(idxA, idxB, idxA+nMemberOfAGroup-2);
            
        }
    }
}


void CGeneticTSPSolver::nextGeneration(void) { // Phenotype Version

    nGeneration++;
    
	int nMemberOfAGroup = nPopulation / nNumberOfGroups;

	for (int g= 0; g<nNumberOfGroups; g++) {

		int start = g*nMemberOfAGroup;
		int end = (g == nNumberOfGroups - 1) ? nPopulation : start + nMemberOfAGroup;

        // competition
        debegMessage("competition group %d: (%d - %d)\n", g, start, end-1);
        nMemberOfAGroup = end - start;
        for (int i = 0; i<nMemberOfAGroup/2; i++) {
            int idxA = start+i*2;
            int idxB = idxA+1;
            int winner = mFitness[idxA]<mFitness[idxB]?idxA:idxB;
            copyGene(winner, i+start);
            debegMessage("%d %d : winner: %d stored at %d (%d)\n", idxA, idxB, winner, i, mFitness[i]);
        }
        if (nMemberOfAGroup%2) {
            copyGene(end-1, start+nMemberOfAGroup/2);
            debegMessage("last one at %d moved to %d\n", end-1, start+nMemberOfAGroup/2);
        }
        ///////////////////////
        
        // crossover
        for (int i = 0; i<(nMemberOfAGroup+1)/2; i+=2) {
            int idxA = start+i;
            int idxB = idxA+1;
            //int child = start+(nMemberOfAGroup+1)/2 + i/2;
            //if (rand()%2) crossoverBCSCX(idxA, idxB, child);
            //else crossoverABCSCX(idxA, idxB, child);
            
            int child = start+(nMemberOfAGroup+1)/2 + i;
            crossoverBCSCX(idxA, idxB, child);
            crossoverABCSCX(idxA, idxB, child+1);
            
            debegMessage("cross: %d %d -> %d\n", idxA, idxB, child);
        }
        ///////////////////////
        
        
        // mutation
        
        for (int i = 0; i<nMemberOfAGroup/4; i++) {
            
            debegMessage("mutate %d -> %d\n", start + i, start + i + (nMemberOfAGroup+1)/2 + (nMemberOfAGroup+1)/4);
            mutate(start + i, start + i + (nMemberOfAGroup+1)/2 + (nMemberOfAGroup+1)/4);
        }
        
        
        
        intergroupMarriage();
       
	}

}
