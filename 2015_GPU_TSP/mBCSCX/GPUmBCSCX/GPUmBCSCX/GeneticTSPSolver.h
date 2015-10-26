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
    float Temperature;
    
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
    void crossoverBCSCX(int idxA, int idxB, int idxC);
    void crossoverABCSCX(int idxA, int idxB, int idxC);
    void mutate(int parent, int mutChild);

	
    int getLegitimateNodeBCSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx);
    

    
public:
	CGeneticTSPSolver();
    CGeneticTSPSolver(CCityLocData *inputData, int nGenes, int nGroups);
	~CGeneticTSPSolver();

	void LoadData(CCityLocData *inputData, int nGenes, int nGroups);
    void LoadSolution(char *fname);
	void RemoveData();
    void fixGene(int idx);

    void initSolver(void);
    
    void printGeneAndFitness(void);
    void drawGene(float dx, float dy, float scaleX = 1.0, float scaleY = 1.0);
    
    void computeFitnessOf(int idx);
    void computeFitness(void) ;
    void intergroupMarriage(void);
    
	void nextGeneration(void);
    
    void copySolution(int *SolutionCpy);
    
    float getBestFitness(void) { return bestFitness; }
    int   getGeneration(void) { return nGeneration; }
    int   getNumCities(void) { return nCities; }
    int   getBestGeneIdx(void) { return bestGeneIdx; }
    float getTemerature(void) { return Temperature; }
};
#endif /* defined(__TSP__GeneticTSPSolver__) */
