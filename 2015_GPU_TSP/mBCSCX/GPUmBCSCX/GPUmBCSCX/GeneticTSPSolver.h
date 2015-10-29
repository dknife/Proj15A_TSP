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

enum CROSSOVERMETHOD {
    BCSCX,
    ABCSCX,
    MIXED,
    NUMCROSSOVERMETHODS
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
    
    CCityLocData *cityLocData;
    int nPopulation;
    int nGeneration;
    int nCities;
    int nNumberOfGroups;
    float Temperature;
    int crossoverMethod;
    
    int** gene;
    
    
    int* mFitness;
    int  bestFitness;
    int  bestGeneIdx;
    
public:
    int  fitRecord;
    int* recordHolder;
    bool bHeating;
    int  nCycleGeneration;
    bool recordBroken;
    
	
	
private:
    void shuffleGene(int idx, int nShuffle);
//    void reverseGene(int idx);
    void swapGene(int idxA, int idxB); // swap A and B (phenotype)
    void copyGene(int idxA, int idxB); // copy A to  B (phenotype)
    
    // <crossover of A and B> is stored into C
    void crossoverBCSCX(int idxA, int idxB, int idxC);
    void crossoverABCSCX(int idxA, int idxB, int idxC);
    void mutate(int parent, int mutChild);
    void invertGene(int idx, int start, int end);
	
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
    void intergroupMarriage(int groupIdx);
    
	void nextGeneration(void);
    
    void copySolution(int *SolutionCpy);
    void copyRecordHolder(int *SolutionCpy);
    
    float getBestFitness(void) { return bestFitness; }
    int   getGeneration(void) { return nGeneration; }
    int   getNumCities(void) { return nCities; }
    int   getBestGeneIdx(void) { return bestGeneIdx; }
    float getTemerature(void) { return Temperature; }
    int   getFitRecord(void) { return fitRecord; }
    
    void  setTemperature(float Temp) { Temperature = Temp; }
    void  changeCrossoverMethod(void) { crossoverMethod = (crossoverMethod+1)%CROSSOVERMETHOD::NUMCROSSOVERMETHODS; }
    const char *getCrossoverMethod(void) {
        switch (crossoverMethod) {
            case CROSSOVERMETHOD::BCSCX: return "BCSCX";
            case CROSSOVERMETHOD::ABCSCX: return "ABCSCX";
            case CROSSOVERMETHOD::MIXED: return "MIXED";
            default: return "Invalid";
        }
    }
};
#endif /* defined(__TSP__GeneticTSPSolver__) */
