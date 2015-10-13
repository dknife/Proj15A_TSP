//
//  AdjacentMatrixGen.cpp
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#include <iomanip>
#include <math.h>

#include "AdjacentMatrixGen.h"
using namespace std;


float CAdjacentMatrixGen::cityDistance(int i, int j) {
	float dx = cityLoc[i].x - cityLoc[j].x;
	float dy = cityLoc[i].y - cityLoc[j].y;
    return sqrt(dx*dx+dy*dy);
}
CAdjacentMatrixGen::CAdjacentMatrixGen(int numRows, int numCols) : nRows(numRows), nCols(numCols) {
    
	cityLoc = new Point[nRows];

	for(int i=0;i<nRows;i++) {
		cityLoc[i].x = rangeRandomf(0.0, 0.4)* ((rand()%2)?1:-1);
		cityLoc[i].y = rangeRandomf(0.0, 0.4)* ((rand()%2)?1:-1);        
	}

    matrix = new float*[numRows];
    for (int i=0; i<nRows; i++) {
        matrix[i] = new float[nCols];
    }

    for (int i=0; i<nRows; i++) {
        for(int j=i; j<nCols; j++) {
            float distance = cityDistance(i, j);
            matrix[i][j] = (i==j)?0.0f:distance;
			matrix[j][i] = (i==j)?0.0f:distance; //rangeRandomf(0.1, 1.0);
        }
    }
}


Proxy CAdjacentMatrixGen::operator[] (int idx) {
    return (idx<nRows&&idx>=0)?Proxy(matrix[idx]):Proxy(matrix[-1]);
}

Point CAdjacentMatrixGen::getLocation(int cityIdx) {
	return this->cityLoc[cityIdx];
}

void CAdjacentMatrixGen::printMatrix(void) {
    for (int i=0; i<nRows; i++) {
        for(int j=0; j<nCols; j++) {
            cout<<fixed << setw(6)<<(*this)[i][j]<<" ";
        }
        cout<<endl;
    }
}
