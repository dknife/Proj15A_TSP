
//
//  GeneticTSPSolver.cpp
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#include "GeneticTSPSolver.h"
#include <math.h>
#ifdef WIN32
#include <windows.h>
//#include <gl/glew.h>
#include <gl/gl.h>
#include <gl/glut.h>
#else // MAC OS X
#include <OpenGL/OpenGL.h>
#include <GLUT/GLUT.h> // OpenGL utility toolkit
#endif

CCitySearchIndex::CCitySearchIndex(int numberOfCities) {
	nSize = numberOfCities;
	validity = new bool[numberOfCities];
	forwardJump = new int[numberOfCities];
	backwardJump = new int[numberOfCities];
	for (int i = 0; i<numberOfCities; i++) {
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
	if (updateFowardIdx < 0) updateFowardIdx += nSize;
	forwardJump[updateFowardIdx] += forwardJump[i];
	int updateBackwardIdx = (i + forwardJump[i]) % nSize;
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
	nCycleGeneration = 100;

	nCities = 0;
	mFitness = NULL;

	gene = NULL;
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
	if (gene) {
		for (int i = 0; i<nPopulation; i++) {
			delete[] gene[i];
		}
		delete[] gene;
	}
	if (mFitness) delete[] mFitness;

	//////////////////

	cityLocData = inputData;
	nPopulation = nGenes;
	nNumberOfGroups = nGroups;

	nCities = cityLocData->numCities;

	mFitness = new int[nPopulation];


	gene = new int*[nPopulation];
	recordHolder = new int[nCities];

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
		gene[0][i] = cityId - 1;
	}
	computeFitnessOf(0);

	printf("solution loaded\n");
}


void CGeneticTSPSolver::initSolver(void) {
	nGeneration = 0;
	Temperature = 100.0;
	crossoverMethod = CROSSOVERMETHOD::BCSCX;
	recordBroken = false;
	bHeating = false;

	for (int i = 0; i<nPopulation; i++) {
		gene[i] = new int[nCities];
	}
	for (int i = 0; i<nPopulation; i++) {
		for (int j = 0; j<nCities; j++) gene[i][j] = j;
	}
	for (int i = 0; i<nPopulation; i++) {
		shuffleGene(i, nCities / 2);
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
	for (int i = 1; i<nShuffle; i++) {
		idxA = rangeRandomi(1, nCities - 1);
		idxB = rangeRandomi(1, nCities - 1);

		swap(gene[idx][idxA], gene[idx][idxB]);
	}
}

void CGeneticTSPSolver::swapGene(int idxA, int idxB) {
	for (int i = 0; i<nCities; i++) swap(gene[idxA][i], gene[idxB][i]);

	swap(mFitness[idxA], mFitness[idxB]);
}

void CGeneticTSPSolver::copyGene(int idxA, int idxB) {
	for (int i = 0; i<nCities; i++) {
		gene[idxB][i] = gene[idxA][i];
	}
	mFitness[idxB] = mFitness[idxA];
}

int CGeneticTSPSolver::getLegitimateNodeBCSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx) {
	int idx = orderOfCity[curCity];
	int forward = (idx + citySearchIdx.getForwardJump(idx)) % nCities;
	int backward = idx - citySearchIdx.getBackwardJump(idx);
	while (backward<0) backward += nCities;

	int nCity1 = cityTour[forward];
	int nCity2 = cityTour[backward];
	int nextCity = (cityLocData->cityDistance(curCity, nCity1) < cityLocData->cityDistance(curCity, nCity2)) ? nCity1 : nCity2;
	return nextCity;


}

void CGeneticTSPSolver::invertGene(int idx, int start, int end) {
	int half = (end - start + 1) / 2;
	for (int i = 0; i<half; i++) {
		swap(gene[idx][start + i], gene[idx][end - i]);
	}

}

void CGeneticTSPSolver::mutate(int parent, int idx) {

	copyGene(parent, idx);

	// inverting a section


	int idxA;
	int idxB;
	int idxMid;
	idxA = rangeRandomi(0, nCities - 1);
	idxB = rangeRandomi(0, nCities - 1);
	if (idxA>idxB) { int T = idxA; idxA = idxB; idxB = T; }

	//invertGene(idx, idxA+1, idxB-1);

	int rV = rand() % 3;
	int half, temp;
	int dMax, target;
	switch (rV) {
	case 0:
		invertGene(idx, idxA + 1, idxB - 1);
		break;
	case 1:
		if (idxA == 0 || idxA == idxB) break;
		target = idxA;
		dMax = 0;
		for (int i = idxB - 1; i<idxA; i--) {
			int d1 = cityLocData->cityDistance(gene[idx][idxA - 1], gene[idx][idxA]) + cityLocData->cityDistance(gene[idx][idxA], gene[idx][idxA + 1]) + cityLocData->cityDistance(gene[idx][i], gene[idx][i + 1]);
			int d2 = cityLocData->cityDistance(gene[idx][idxA - 1], gene[idx][idxA + 1]) + cityLocData->cityDistance(gene[idx][i], gene[idx][idxA]) + cityLocData->cityDistance(gene[idx][idxA], gene[idx][i + 1]);
			if (d1 - d2 > dMax) {
				dMax = d1 - d2;
				target = i;
			}
		}
		temp = gene[idx][idxA];
		for (int i = idxA; i<target; i++) {
			gene[idx][i] = gene[idx][i + 1];
		}
		gene[idx][target] = temp;
		break;
	case 2:
		if (idxB - idxA < 6) break;
		idxMid = idxA + rangeRandomi(3, idxB - idxA - 3);

		invertGene(idx, idxA + 1, idxMid);
		invertGene(idx, idxMid + 1, idxB - 1);
		break;

	default:
		break;
	}


	computeFitnessOf(idx);
}


void CGeneticTSPSolver::fixGene(int idx) {

	int idxA;
	int idxB;
	int search = 0;
	int maxSearch = (int)(sqrt(nCities) + 0.5);
	//maxSearch*=10;
	//if(maxSearch>nCities-2) maxSearch=nCities-2;
	idxA = rangeRandomi(0, nCities - 1 - maxSearch);
	idxB = rangeRandomi(0, nCities - 1 - maxSearch);

	int distOrg, distNew;
	int maxGain;
	int iForMaxGain = -1, jForMaxGain = -1;
	bool bIntersectFound;
	bool bMovableFound;


	// Start: city move -----------------------------------
	maxGain = 0;
	bMovableFound = false;
	idxA = rangeRandomi(1, nCities - 1 - maxSearch);
	idxB = rangeRandomi(1, nCities - 1 - maxSearch);
	int Prev1, Next1, Prev2, Next2;

	for (int i = 0; i<maxSearch; i++) {
		for (int j = 0; j<maxSearch; j++) {

			int v1 = idxA + i;
			int v2 = idxB + j;
			Prev1 = v1 - 1;
			Next1 = v1 + 1;
			Prev2 = v2 - 1;
			Next2 = v2 + 1;

			distOrg = cityLocData->cityDistance(gene[idx][Prev1], gene[idx][v1]) + cityLocData->cityDistance(gene[idx][v1], gene[idx][Next1]) + cityLocData->cityDistance(gene[idx][v2], gene[idx][Next2]);
			distNew = cityLocData->cityDistance(gene[idx][Prev1], gene[idx][Next1]) + cityLocData->cityDistance(gene[idx][v2], gene[idx][v1]) + cityLocData->cityDistance(gene[idx][v1], gene[idx][Next2]);
			int gain = distOrg - distNew;
			if (gain>maxGain) {
				bMovableFound = true;
				iForMaxGain = v1;
				jForMaxGain = v2;
				maxGain = gain;
			}

		}
	}

	bMovableFound = false;
	if (bMovableFound) {
		//if(iForMaxGain > jForMaxGain) { int t = iForMaxGain; iForMaxGain=jForMaxGain; jForMaxGain=t; }
		int cityBackup = gene[idx][iForMaxGain];
		for (int i = iForMaxGain; i != jForMaxGain; i = (i + 1) % nCities) {
			swap(gene[idx][i%nCities], gene[idx][(i + 1) % nCities]);
		}
		gene[idx][jForMaxGain] = cityBackup;
	}
	// End: city move -----------------------------------



	maxGain = 0;
	bIntersectFound = false;

	for (int i = 0; i<maxSearch; i++) {
		for (int j = 0; j<maxSearch; j++) {
			int v1 = idxA + i;
			int v2 = idxB + j;
			if (v1 == v2) continue;
			distOrg = cityLocData->cityDistance(gene[idx][v1], gene[idx][v1 + 1]) + cityLocData->cityDistance(gene[idx][v2], gene[idx][v2 + 1]);
			distNew = cityLocData->cityDistance(gene[idx][v1], gene[idx][v2]) + cityLocData->cityDistance(gene[idx][v1 + 1], gene[idx][v2 + 1]);
			int gain = distOrg - distNew;
			if (gain>maxGain) {
				bIntersectFound = true;
				iForMaxGain = v1;
				jForMaxGain = v2;
				maxGain = gain;
			}
		}
	}

	if (bIntersectFound) {
		if (iForMaxGain > jForMaxGain) { int t = iForMaxGain; iForMaxGain = jForMaxGain; jForMaxGain = t; }
		int half = (jForMaxGain - iForMaxGain) / 2;
		for (int i = 0; i<half; i++) {
			swap(gene[idx][iForMaxGain + 1 + i], gene[idx][jForMaxGain - i]);
		}

	}



	maxGain = 0;
	bIntersectFound = false;

	for (int i = 0; i<nCities - 1; i++) {
		for (int j = 0; j<maxSearch; j++) {
			int v1 = i;
			int v2;
			int gain;

			v2 = (i + 2 + j) % nCities;
			if (v2 != 0) {
				distOrg = cityLocData->cityDistance(gene[idx][v1], gene[idx][v1 + 1]) + cityLocData->cityDistance(gene[idx][v2], gene[idx][(v2 + 1) % nCities]);
				distNew = cityLocData->cityDistance(gene[idx][v1], gene[idx][v2]) + cityLocData->cityDistance(gene[idx][v1 + 1], gene[idx][(v2 + 1) % nCities]);
				gain = distOrg - distNew;
				if (gain>maxGain) {
					bIntersectFound = true;
					maxGain = gain;
					iForMaxGain = v1;
					jForMaxGain = v2;
				}
			}

			v2 = i - 2 - j;
			if (v2<0) v2 += nCities;

			if (v2 != 0) {
				distOrg = cityLocData->cityDistance(gene[idx][v1], gene[idx][v1 + 1]) + cityLocData->cityDistance(gene[idx][v2], gene[idx][(v2 + 1) % nCities]);
				distNew = cityLocData->cityDistance(gene[idx][v1], gene[idx][v2]) + cityLocData->cityDistance(gene[idx][v1 + 1], gene[idx][(v2 + 1) % nCities]);
				gain = distOrg - distNew;
				if (gain>maxGain) {
					bIntersectFound = true;
					maxGain = gain;
					iForMaxGain = v1;
					jForMaxGain = v2;
				}
			}




		}
	}


	if (bIntersectFound) {
		if (iForMaxGain > jForMaxGain) { int t = iForMaxGain; iForMaxGain = jForMaxGain; jForMaxGain = t; }
		int half = (jForMaxGain - iForMaxGain) / 2;
		for (int i = 0; i<half; i++) {
			swap(gene[idx][iForMaxGain + 1 + i], gene[idx][jForMaxGain - i]);
		}

	}


	computeFitnessOf(idx);
}


void CGeneticTSPSolver::crossoverBCSCX(int idxA, int idxB, int idxC) {


	int candidate1, candidate2;
	int curCity = 0;
	int nextCity;
	bool *cityValid = new bool[nCities];
	int  *crossover = new int[nCities];
	crossover[0] = 0;


	for (int i = 1; i<nCities; i++) {
		cityValid[i] = true;
	}	cityValid[0] = false;

	CCitySearchIndex citySearchIdx_A(nCities);
	CCitySearchIndex citySearchIdx_B(nCities);
	citySearchIdx_A.invalidate(0);
	citySearchIdx_B.invalidate(0);

	int *orderOfCity_A = new int[nCities];
	int *orderOfCity_B = new int[nCities];

	for (int i = 0; i<nCities; i++) {
		int city;
		city = gene[idxA][i]; orderOfCity_A[city] = i;
		city = gene[idxB][i]; orderOfCity_B[city] = i;
	}

	for (int i = 1; i<nCities; i++) {
		candidate1 = getLegitimateNodeBCSCX(curCity, gene[idxA], orderOfCity_A, citySearchIdx_A);
		candidate2 = getLegitimateNodeBCSCX(curCity, gene[idxB], orderOfCity_B, citySearchIdx_B);
		if (cityLocData->cityDistance(curCity, candidate1) < cityLocData->cityDistance(curCity, candidate2))
			nextCity = candidate1;
		else nextCity = candidate2;
		crossover[i] = nextCity;
		cityValid[nextCity] = false;
		curCity = nextCity;
		citySearchIdx_A.invalidate(orderOfCity_A[nextCity]);
		citySearchIdx_B.invalidate(orderOfCity_B[nextCity]);
	}

	for (int i = 0; i<nCities; i++) {
		gene[idxC][i] = crossover[i];
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


	for (int i = 1; i<nCities; i++) {
		cityValid[i] = true;
	}	cityValid[0] = false;

	CCitySearchIndex citySearchIdx_A(nCities);
	CCitySearchIndex citySearchIdx_B(nCities);
	citySearchIdx_A.invalidate(0);
	citySearchIdx_B.invalidate(0);

	int *orderOfCity_A = new int[nCities];
	int *orderOfCity_B = new int[nCities];

	for (int i = 0; i<nCities; i++) {
		int city;
		city = gene[idxA][i]; orderOfCity_A[city] = i;
		city = gene[idxB][i]; orderOfCity_B[city] = i;
	}

	for (int i = 1; i<nCities; i++) {
		if (bUse1stGene) {
			nextCity = getLegitimateNodeBCSCX(curCity, gene[idxA], orderOfCity_A, citySearchIdx_A);
		}
		else {
			nextCity = getLegitimateNodeBCSCX(curCity, gene[idxB], orderOfCity_B, citySearchIdx_B);
		}
		crossover[i] = nextCity;
		cityValid[nextCity] = false;
		curCity = nextCity;
		citySearchIdx_A.invalidate(orderOfCity_A[nextCity]);
		citySearchIdx_B.invalidate(orderOfCity_B[nextCity]);
		bUse1stGene = bUse1stGene ? false : true;
	}

	for (int i = 0; i<nCities; i++) {
		gene[idxC][i] = crossover[i];
	}
	delete[] cityValid;
	delete[] crossover;

	delete[] orderOfCity_A;
	delete[] orderOfCity_B;



	computeFitnessOf(idxC);
}




void CGeneticTSPSolver::copySolution(int *SolutionCpy) {
	if (!SolutionCpy) return;

	for (int i = 0; i<nCities; i++) {
		SolutionCpy[i] = gene[bestGeneIdx][i];
	}
}

void CGeneticTSPSolver::copyRecordHolder(int *SolutionCpy) {
	if (!SolutionCpy) return;

	for (int i = 0; i<nCities; i++) {
		SolutionCpy[i] = recordHolder[i];
	}
}


void CGeneticTSPSolver::computeFitnessOf(int idx) {

	mFitness[idx] = 0;
	for (int j = 0; j<nCities; j++) {
		mFitness[idx] += cityLocData->cityDistance(gene[idx][j], gene[idx][(j + 1) % nCities]);
	}

}

void CGeneticTSPSolver::computeFitness(void) {

	int nMemberOfAGroup = nPopulation / nNumberOfGroups;
	recordBroken = false;

	for (int g = 0; g<nNumberOfGroups; g++) { // for every local group

		int start = g*nMemberOfAGroup;
		int end = (g == nNumberOfGroups - 1) ? nPopulation : start + nMemberOfAGroup;

		int localBestFitness = mFitness[start];
		int   localBestGeneIdx = start;

		for (int i = start; i<end; i++) {
			computeFitnessOf(i);
			if (mFitness[i]<localBestFitness) { localBestFitness = mFitness[i]; localBestGeneIdx = i; }
			debegMessage("fit (%d) = %d\n", i, mFitness[i]);
		}

		// move the best in the local group to be the first gene in the group
		debegMessage("group(%d) - swapping %d (%d) and %d (%d)\n", g, start, mFitness[start], localBestGeneIdx, localBestFitness);
		swapGene(start, localBestGeneIdx);

		//fixGene(start);
		//computeFitnessOf(start);

	}



	bestFitness = mFitness[0];
	bestGeneIdx = 0;


	// find the best in the total population
	for (int i = 0; i<nPopulation; i += nMemberOfAGroup) {
		if (mFitness[i]<bestFitness) { bestGeneIdx = i; bestFitness = mFitness[i]; }
	}


	debegMessage("bestGene = %d (%d)\n", bestGeneIdx, bestFitness);

	if (nGeneration <= 1) {
		fitRecord = mFitness[bestGeneIdx];
		for (int i = 0; i<nCities; i++) {
			recordHolder[i] = gene[bestGeneIdx][i];
		}
	}
	else {
		if (mFitness[bestGeneIdx]<fitRecord) {
			fitRecord = mFitness[bestGeneIdx];
			for (int i = 0; i<nCities; i++) {
				recordHolder[i] = gene[bestGeneIdx][i];
			}
			recordBroken = true;
		}
		else {
			recordBroken = false;
		}
	}

}

void CGeneticTSPSolver::intergroupMarriage(int groupIdx) {

    if(nNumberOfGroups<2) return;
    
	int nMemberOfAGroup = nPopulation / nNumberOfGroups;
	int idxA = groupIdx*nMemberOfAGroup;
	int g = groupIdx;
	while (g == groupIdx) g = rangeRandomi(0, nNumberOfGroups - 1);
	int idxB = g*nMemberOfAGroup;
	/*
	int idxB = ((groupIdx+1)%nNumberOfGroups)*nMemberOfAGroup;
	int idxC = ((groupIdx-1+nNumberOfGroups)%nNumberOfGroups)*nMemberOfAGroup;
	*/
	switch (crossoverMethod) {

	case CROSSOVERMETHOD::BCSCX:
		crossoverBCSCX(idxA, idxB, idxA + nMemberOfAGroup / 2 - 1);
		break;
	case CROSSOVERMETHOD::ABCSCX:
		crossoverABCSCX(idxA, idxB, idxA + nMemberOfAGroup / 2 - 1);
		break;
	case CROSSOVERMETHOD::MIXED:
		if (rand() % 2) crossoverBCSCX(idxA, idxB, idxA + nMemberOfAGroup / 2 - 1);
		else crossoverABCSCX(idxA, idxB, idxA + nMemberOfAGroup / 2 - 2);
		break;
	}


}


void CGeneticTSPSolver::nextGeneration(void) { // Phenotype Version

	nGeneration++;


	int nMemberOfAGroup = nPopulation / nNumberOfGroups;



	// linear cooling
	float cooling = 100.0 / nCycleGeneration;
	float prevTemp = Temperature;
	Temperature -= cooling;
	if (Temperature<0) Temperature = 100.0;

	// cosine temperature
	//Temperature = 100.0 * 0.5 * (1.0+ cos(2.0*3.14*nGeneration/float(nCycleGeneration) ) );


	for (int g = 0; g<nNumberOfGroups; g++) {

		int start = g*nMemberOfAGroup;
		int end = (g == nNumberOfGroups - 1) ? nPopulation : start + nMemberOfAGroup;

		// competition
		debegMessage("competition group %d: (%d - %d)\n", g, start, end - 1);
		nMemberOfAGroup = end - start;
		for (int i = 0; i<nMemberOfAGroup / 2; i++) {
			int idxA = start + i * 2;
			int idxB = idxA + 1;
			int winner = mFitness[idxA]<mFitness[idxB] ? idxA : idxB;
			int loser = winner == idxA ? idxB : idxA;

			if (bHeating && rangeRandomf(0.0, 100.0) < Temperature / 2.5) winner = loser;


			copyGene(winner, i + start);
			debegMessage("%d %d : winner: %d stored at %d (%d)\n", idxA, idxB, winner, i, mFitness[i]);
		}
		if (nMemberOfAGroup % 2) {
			copyGene(end - 1, start + nMemberOfAGroup / 2);
			debegMessage("last one at %d moved to %d\n", end - 1, start + nMemberOfAGroup / 2);
		}
		///////////////////////

		// crossover (66.6%) or mutation (33%)
		for (int i = 0; i<nMemberOfAGroup / 2; i++) {
			int idxA = start + i;
			int idxB = idxA + 1;

			int child = start + nMemberOfAGroup / 2 + i;

			// mutation rate = 33%
			int iM3 = i % 3;

			if (iM3 == 0) {
				mutate(idxA, child);
			}
			else {
				if (crossoverMethod == CROSSOVERMETHOD::BCSCX) crossoverBCSCX(idxA, idxB, child);
				else if (crossoverMethod == CROSSOVERMETHOD::ABCSCX) crossoverABCSCX(idxA, idxB, child);
				else {
					if (rand() % 2) crossoverABCSCX(idxA, idxB, child);
					else crossoverBCSCX(idxA, idxB, child);
				}
			}
			debegMessage("cross: %d %d -> %d\n", idxA, idxB, child);
		}
		///////////////////////

		fixGene(start);


		intergroupMarriage(g);

	}


}




void CGeneticTSPSolver::drawGene(float intervalX, float intervalY, float scaleX, float scaleY) {



	for (int i = 0; i<nPopulation; i++) {
		glBegin(GL_LINE_STRIP);
		for (int j = 0; j<nCities; j++) {
			int city = gene[i][j];
			float color = city / ((float)nCities);
			glColor3f(color, color, 1);
			glVertex2d(j*scaleX + intervalX, i*scaleY + intervalY);
		}
		glEnd();
	}


}
