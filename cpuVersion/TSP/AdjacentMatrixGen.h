//
//  AdjacentMatrixGen.h
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#ifndef __TSP__AdjacentMatrixGen__
#define __TSP__AdjacentMatrixGen__

#include <iostream>
#include <stdlib.h>
#include "utilities.h"

struct Point {
    float x;
	float y;    
};

class Proxy {
    float* _array;
public:
    Proxy(float* array) : _array(array) {}
    float operator[] (int idx) { return (idx>=0)?_array[idx]:-1.0f; }
};

class CAdjacentMatrixGen {

    float **matrix;
    Point *cityLoc;
	
public:
    int  nRows;
    int  nCols;
    
// methods
public:
    CAdjacentMatrixGen(int numRows, int numCols) ;
    
    Proxy operator[] (int idx);
	Point getLocation(int cityIdx);
	float cityDistance(int i, int j);
    void  printMatrix(void);
};

#endif /* defined(__TSP__AdjacentMatrixGen__) */
