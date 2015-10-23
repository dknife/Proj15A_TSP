//
//  GeneticTSPSolver.h
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#ifndef __TSP__GeneticTSPSolver__
#define __TSP__GeneticTSPSolver__

#include <iostream>
#include <iomanip>
#include "cityLocData.h"
#include "utilities.h"


using namespace std;

class CCitySearchIndex {
    int nSize;
    int  *forwardJump;
    int  *backwardJump;
public:
    bool *validity;
public:
    CCitySearchIndex(int numberOfCities);
    ~CCitySearchIndex();
    void invalidate(int i);
    int getForwardJump(int i);
    int getBackwardJump(int i);
};

class CGeneticTSPSolver {
    
    int nPopulation;
    int nGeneration;
    int nCities;
    int nNumberOfGroups;
    
    int** gene;
    
    int* mFitness;
    int  bestFitness;
    int   bestGeneIdx;
    
	CCityLocData *cityLocData;
	
    void shuffleGene(int idx, int nShuffle);
//    void reverseGene(int idx);
    void swapGene(int idxA, int idxB); // swap A and B (phenotype)
    void copyGene(int idxA, int idxB); // copy A to  B (phenotype)
    
    // <crossover of A and B> is stored into C
    void crossover(int idxA, int idxB, int idxC);
    void mutate(int parent, int mutChild);
	
    int getLegitimateNodeBCSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx);
    

    
public:
	CGeneticTSPSolver();
    CGeneticTSPSolver(CCityLocData *inputData, int nGenes, int nGroups);
	~CGeneticTSPSolver();

	void LoadData(CCityLocData *inputData, int nGenes, int nGroups);
    void LoadSolution(char *fname);
	void RemoveData();

    void initSolver(void);
    
    void printGeneAndFitness(void);
    void printGene(int idx);
    void computeFitnessOf(int idx);
    void computeFitness(void) ;
    
	void nextGeneration(void);
    
    void copySolution(int *SolutionCpy);
    
    float getBestFitness(void) { return bestFitness; }
    int   getGeneration(void) { return nGeneration; }
};
#endif /* defined(__TSP__GeneticTSPSolver__) */
