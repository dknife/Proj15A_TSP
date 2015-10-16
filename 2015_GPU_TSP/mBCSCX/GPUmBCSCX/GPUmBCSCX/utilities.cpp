//
//  utilities.cpp
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#include "utilities.h"

float rangeRandomf(float min, float max) {
    return min + (max-min)*(float(rand())/float(RAND_MAX));
}

int rangeRandomi(int min, int max) {
    return min + rand()%(max-min+1);
}

void  swap(int &a, int &b) {
    int t=a; a = b; b = t;
}

