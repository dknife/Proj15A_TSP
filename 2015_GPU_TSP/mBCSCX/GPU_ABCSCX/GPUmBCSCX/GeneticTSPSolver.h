//
//  GeneticTSPsolver->h
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
    virtual ~CCitySearchIndex();
    void invalidate(int i);
    int getForwardJump(int i);
    int getBackwardJump(int i);
};

class CGeneticTSPSolver {
    
public:
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
	

    int  fitRecord;
    int* recordHolder;


    
	bool bHeating;
    int  nCycleGeneration;
    bool recordBroken;
    
	
	
private:
    
//    void reverseGene(int idx);
    virtual void swapGene(int idxA, int idxB); // swap A and B (phenotype)
    virtual void copyGene(int idxA, int idxB); // copy A to  B (phenotype)
    
    // <crossover of A and B> is stored into C
    virtual void crossoverBCSCX(int idxA, int idxB, int idxC);
	virtual void crossoverABCSCX(int idxA, int idxB, int idxC);
	virtual void mutate(int parent, int mutChild);
	virtual void invertGene(int idx, int start, int end);
	
	virtual int getLegitimateNodeBCSCX(int curCity, int *cityTour, int *orderOfCity, CCitySearchIndex& citySearchIdx);
    

    
public:
	CGeneticTSPSolver();
	virtual ~CGeneticTSPSolver();

	virtual void LoadData(CCityLocData *inputData, int nGenes, int nGroups);

    virtual void LoadSolution(const char *fname);
	virtual void LoadLocalMinima(const char *fname);
	virtual void RemoveData();
    virtual void fixGene(int idx);

    virtual void initSolver(void);
    
    virtual void drawGene(float dx, float dy, float scaleX = 1.0, float scaleY = 1.0);
    
    virtual void computeFitnessOf(int idx);
    virtual void computeFitness(void) ;
    virtual void intergroupMarriage(int groupIdx);
    
	virtual void nextGeneration(void);
    
    virtual void copySolution(int *SolutionCpy);
    virtual void copyRecordHolder(int *SolutionCpy);
    
	virtual void shuffleGene(int idx, int nShuffle);

	// property get/set methods
    float getBestFitness(void) { return bestFitness; }
    int   getGeneration(void) { return nGeneration; }
    int   getNumCities(void) { return nCities; }
	int   getNumPopulation(void) { return nPopulation; }
    int   getBestGeneIdx(void) { return bestGeneIdx; }
    float getTemerature(void) { return Temperature; }
    int   getFitRecord(void) { return fitRecord; }
	CCityLocData *getCityLocData(void) { return cityLocData; }
    
    void  setTemperature(float Temp) { Temperature = Temp; }
    void  changeCrossoverMethod(void) { crossoverMethod = (crossoverMethod+1)%NUMCROSSOVERMETHODS; }
    const char *getCrossoverMethod(void) {
        switch (crossoverMethod) {
            case BCSCX: return "BCSCX";
            case ABCSCX: return "ARX";
            case MIXED: return "MIXED";
            default: return "Invalid";
        }
    }
};
#endif /* defined(__TSP__GeneticTSPSolver__) */
