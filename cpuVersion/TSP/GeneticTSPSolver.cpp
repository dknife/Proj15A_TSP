
//
//  GeneticTSPSolver.cpp
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#include "GeneticTSPSolver.h"

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

CGeneticTSPSolver::CGeneticTSPSolver(CCityLocData *inputData, int nGenes, int nGroups) {
    cityLocData = inputData;
    nPopulation = nGenes;
    nNumberOfGroups = nGroups;
    
	nCities = cityLocData->numCities;
	mFitness = new float[nPopulation];
    

    mPhenotype = new int*[nPopulation];
    
    initSolver();
    
}

CGeneticTSPSolver::~CGeneticTSPSolver() {
	if (cityLocData) cityLocData = NULL;
	if (mFitness) delete[] mFitness;
	if (mPhenotype) {
		for (int i = 0; i < nPopulation; i++) {
			delete[] mPhenotype[i];
		}
		delete[] mPhenotype;
	}
}

void CGeneticTSPSolver::initSolver(void) {
    nGeneration = 0;
    
    for (int i=0; i<nPopulation; i++) {
        mPhenotype[i] = new int[nCities];
    }
    for (int i=0; i<nPopulation; i++) {
        for (int j=0; j<nCities  ; j++) mPhenotype[i][j] = j;
    }
    for (int i=0; i<nPopulation; i++) {
        shufflePheno(i, nCities/2);
    }
}

void CGeneticTSPSolver::reversePheno(int idx) {
    int max=nCities/2;
    for(int i=1;i<=max;i++) {
        swap(mPhenotype[idx][i], mPhenotype[idx][nCities-i]);
    }
}

void CGeneticTSPSolver::shufflePheno(int idx, int nShuffle) {
    int idxA, idxB;
    for(int i=1;i<nShuffle;i++) {
        idxA = rangeRandomi(1, nCities-1);
        idxB = rangeRandomi(1, nCities-1);
        
        swap(mPhenotype[idx][idxA], mPhenotype[idx][idxB]);
   }
}

void CGeneticTSPSolver::swapPhenotype(int idxA, int idxB) {
    for (int i=0; i<nCities; i++) {
        int T = mPhenotype[idxA][i];
        mPhenotype[idxA][i] = mPhenotype[idxB][i];
        mPhenotype[idxB][i] = T;
    }
}

void CGeneticTSPSolver::copyPhenotype(int idxA, int idxB) {
    for (int i=0; i<nCities; i++) {
        mPhenotype[idxB][i]=mPhenotype[idxA][i];
    }
}

int CGeneticTSPSolver::getLegitimateNodeSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx) {
    
    int idx = orderOfCity[curCity];
    int forward = (idx + citySearchIdx.getForwardJump(idx))%nCities;
    return cityTour[forward];
    /*
    int idx = orderOfCity[curCity];
    int forward = idx + citySearchIdx.getForwardJump(idx);
    if(forward>nCities-1) {
        forward = 0;
        while(citySearchIdx.validity[forward]==false) forward++;
    }
    return cityTour[forward];
     */
}

int CGeneticTSPSolver::getLegitimateNodeTWSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx, bool bForward) {
    int idx = orderOfCity[curCity];
    if(bForward) {
        idx = (idx + citySearchIdx.getForwardJump(idx))%nCities;
    }
    else {
        idx = idx - citySearchIdx.getBackwardJump(idx);
        while(idx<0) idx+=nCities;
    }
    return cityTour[idx];
}

int CGeneticTSPSolver::getLegitimateNodeBCSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx) {
    int idx = orderOfCity[curCity];
    int forward = (idx + citySearchIdx.getForwardJump(idx))%nCities;
    int backward = idx - citySearchIdx.getBackwardJump(idx);
    while(backward<0) backward+=nCities;
    
    int nCity1 = cityTour[forward];
    int nCity2 = cityTour[backward];
    //int nextCity = ((*adjMatrix)[curCity][nCity1] < (*adjMatrix)[curCity][nCity2] )? nCity1: nCity2;
	int nextCity = (cityLocData->cityDistance(curCity, nCity1) < cityLocData->cityDistance(curCity, nCity2)) ? nCity1 : nCity2;
    return nextCity;
    

}

void CGeneticTSPSolver::crossoverPheno(int idxA, int idxB, int idxC, evolutionMode mode, bool bForward) {


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
        city = mPhenotype[idxA][i]; orderOfCity_A[city] = i;
        city = mPhenotype[idxB][i]; orderOfCity_B[city] = i;
    }

    for(int i=1; i<nCities; i++) {
		if(mode == PHENOTYPE_SCX) {
			candidate1 = getLegitimateNodeSCX(curCity, mPhenotype[idxA], orderOfCity_A, citySearchIdx_A);
			candidate2 = getLegitimateNodeSCX(curCity, mPhenotype[idxB], orderOfCity_B, citySearchIdx_B);
			if ( cityLocData->cityDistance(curCity,candidate1)< cityLocData->cityDistance(curCity,candidate2)) 
				nextCity = candidate1;
			else nextCity = candidate2;
		}
        else if(mode == PHENOTYPE_TWSCX) { // PHENOTYPE_TWSCX
			candidate1 = getLegitimateNodeTWSCX(curCity, mPhenotype[idxA], orderOfCity_A, citySearchIdx_A, bForward);
			candidate2 = getLegitimateNodeTWSCX(curCity, mPhenotype[idxB], orderOfCity_B, citySearchIdx_B, bForward);
			if (cityLocData->cityDistance(curCity, candidate1) < cityLocData->cityDistance(curCity, candidate2))
				nextCity = candidate1;
			else nextCity = candidate2;
		}
		else if(mode == PHENOTYPE_BCSCX) { // PHENOTYPE_BCSCX
			candidate1 = getLegitimateNodeBCSCX(curCity, mPhenotype[idxA], orderOfCity_A, citySearchIdx_A);
			candidate2 = getLegitimateNodeBCSCX(curCity, mPhenotype[idxB], orderOfCity_B, citySearchIdx_B);
			if (cityLocData->cityDistance(curCity, candidate1) < cityLocData->cityDistance(curCity, candidate2))
				nextCity = candidate1;
			else nextCity = candidate2;
		}

		
		crossover[i]=nextCity;
		cityValid[nextCity] = false;
		curCity = nextCity;
        citySearchIdx_A.invalidate(orderOfCity_A[nextCity]);
        citySearchIdx_B.invalidate(orderOfCity_B[nextCity]);
	}
    for(int i=0;i<nCities;i++) {
        mPhenotype[idxC][i]=crossover[i];
    }
	delete[] cityValid;
	delete[] crossover;
    
    delete[] orderOfCity_A;
    delete[] orderOfCity_B;
}


void CGeneticTSPSolver::printGeneAndFitness(void) {
        
    for(int i=0;i<nPopulation;i++) {
        mFitness[i] = 0;
        cout<<"Gene "<<setw(3)<<i<<": ";
        for(int j=0;j<nCities;j++) {
            cout<<setw(3)<<mPhenotype[i][j]<<" ";
            mFitness[i] += cityLocData->cityDistance(mPhenotype[i][j], mPhenotype[i][(j+1)%nCities]);
        }
        cout<<endl;
            
        cout<<"fitness = "<<mFitness[i]<<endl;
        
    }
}

void CGeneticTSPSolver::copySolution(int *SolutionCpy) {
    if(!SolutionCpy) return;
    
    for (int i=0; i<nCities; i++) {
        SolutionCpy[i] = mPhenotype[bestGeneIdx][i];
    }
}

void CGeneticTSPSolver::printGene(int idx) {
    
    
    cout<<"City Tour "<<setw(3)<<idx<<": ";
    for(int j=0;j<nCities;j++) {
        cout<<setw(3)<<mPhenotype[idx][j]<<" ";
    }
    cout<<endl;
   
}

void CGeneticTSPSolver::computeFitnessOf(int idx) {
    
    mFitness[idx] = 0;
    for(int j=0;j<nCities;j++) {
        mFitness[idx] += cityLocData->cityDistance( mPhenotype[idx][j] , mPhenotype[idx][(j+1)%nCities] );
    }
    
}

void CGeneticTSPSolver::computeFitness(void) {
    
    int nMemberOfAGroup = nPopulation/nNumberOfGroups;
    
   
    for(int g=0;g<nNumberOfGroups;g++) { // for every local group
        
        int start = g*nMemberOfAGroup;
        int end = (g==nNumberOfGroups-1)?nPopulation: start+nMemberOfAGroup;
        
        float localBestFitness = mFitness[start];
        int   localBestGeneIdx = start;
        
        for(int i=start;i<end;i++) {
            mFitness[i] = 0;
            for(int j=0;j<nCities;j++) {
                mFitness[i] += cityLocData->cityDistance(mPhenotype[i][j] , mPhenotype[i][(j+1)%nCities]);
            }
            if(mFitness[i]<localBestFitness) { localBestFitness = mFitness[i]; localBestGeneIdx = i; }
        }
        
        // move the best in the local group to be the first gene in the group
        swapPhenotype(start, localBestGeneIdx);
        swap(mFitness[start], mFitness[localBestGeneIdx]);
    }
    
    
    // best gene migration
    /*
    if(nNumberOfGroups>1) {
        for(int i=0;i<nNumberOfGroups;i++) {
            int idxA = rangeRandomi(0, nNumberOfGroups-1);
            int idxB = idxA;
            while( idxB == idxA) idxB = nMemberOfAGroup * rangeRandomi(0, nNumberOfGroups-1);
            
            // swapping
            //swapPhenotype(idxA, idxB);
            //swap(mFitness[idxA], mFitness[idxB]);
            
            // best gene copying
            copyPhenotype(idxA, idxB+1);
            mFitness[idxB+1] = mFitness[idxA];
            copyPhenotype(idxB, idxA+1);
            mFitness[idxA+1] = mFitness[idxB];
            
        }
    }*/
    
    bestFitness = mFitness[0];
    bestGeneIdx = 0;
    
    // find the best in the total population
    for(int i=0;i<nPopulation;i+=nMemberOfAGroup) {
        if(mFitness[i]<bestFitness) { bestGeneIdx = i; bestFitness = mFitness[i]; }
    }
    
}
    
void CGeneticTSPSolver::nextGenerationWithPhenotype(evolutionMode mode) {
    
    int nMemberOfAGroup = nPopulation/nNumberOfGroups;
    
    for(int g=0;g<nNumberOfGroups;g++) {
        
        int start = g*nMemberOfAGroup;
        int end = (g==nNumberOfGroups-1)?nPopulation: start+nMemberOfAGroup;
        
        int winnerA, winnerB, loserA, loserB;
        for(int i=start;i<end;i++) {
            int idxA = rangeRandomi(start, end-1);
            int idxB = idxA;
            while( idxB == idxA) idxB = rangeRandomi(start, end-1);
            
            if(mFitness[idxA]<mFitness[idxB]) {
                winnerA = idxA; loserA = idxB;
            }
            else {
                winnerA = idxB; loserA = idxA;
            }
            
            idxA = rangeRandomi(start, end-1);
            idxB = idxA;
            while( idxB == idxA) idxB = rangeRandomi(start, end-1);
            
            if(mFitness[idxA]<mFitness[idxB]) {
                winnerB = idxA; loserB = idxB;
            }
            else {
                winnerB = idxB; loserB = idxA;
            }
            
            crossoverPheno(winnerA, winnerB, loserA, mode, true);
            computeFitnessOf(loserA);
            
            if(PHENOTYPE_TWSCX)crossoverPheno(winnerA, winnerB, loserB, mode, false);
            else shufflePheno(loserB, rangeRandomi(nCities/3, nCities/2) );
            //else reversePheno(loserB );
            computeFitnessOf(loserB);
        }

    }
    

    
}


void CGeneticTSPSolver::nextGeneration(evolutionMode mode) { // Phenotype Version

    nGeneration++;
    
    switch (mode) {
        case PHENOTYPE_SCX:
            nextGenerationWithPhenotype(PHENOTYPE_SCX);
            break;
        case PHENOTYPE_BCSCX:
            nextGenerationWithPhenotype(PHENOTYPE_BCSCX);
            break;
        case PHENOTYPE_TWSCX:
            nextGenerationWithPhenotype(PHENOTYPE_TWSCX);
            break;
        default:
            break;
    }

}
