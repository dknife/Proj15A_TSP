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
#include "AdjacentMatrixGen.h"
#include "utilities.h"

enum evolutionMode {
    PHENOTYPE_SCX,
    PHENOTYPE_BCSCX,
    PHENOTYPE_TWSCX
};
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
    
    int** mPhenotype;
    
    float* mFitness;
    float bestFitness;
    int   bestGeneIdx;
    
    CAdjacentMatrixGen *adjMatrix;
    
    void shufflePheno(int idx, int nShuffle);
    void reversePheno(int idx);
    void swapPhenotype(int idxA, int idxB); // swap A and B (phenotype)
    void copyPhenotype(int idxA, int idxB); // copy A to  B (phenotype)
    
    // <crossover of A pheno and B pheno> is stored into C
    void crossoverPheno(int idxA, int idxB, int idxC, evolutionMode mode, bool bForward=true);
	
    int getLegitimateNodeBCSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx);
    int getLegitimateNodeSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx);
    int getLegitimateNodeTWSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx, bool bForward=true);
    
    void nextGenerationWithPhenotype(evolutionMode mode);
	
    
public:
    CGeneticTSPSolver(CAdjacentMatrixGen &adjMat, int nGenes, int nGroups);
    void initSolver(void);
    
    void printGeneAndFitness(void);
    void printGene(int idx);
    void computeFitnessOf(int idx);
    void computeFitness(void) ;
    
    void nextGeneration(evolutionMode mode);
    
    void copySolution(int *SolutionCpy);
    
    float getBestFitness(void) { return bestFitness; }
    int   getGeneration(void) { return nGeneration; }
};
#endif /* defined(__TSP__GeneticTSPSolver__) */
