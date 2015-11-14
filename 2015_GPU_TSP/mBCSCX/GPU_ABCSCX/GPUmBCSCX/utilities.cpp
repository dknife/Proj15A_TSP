//
//  utilities.cpp
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#include <stdlib.h>
#include "utilities.h"

#define NDEBUG // NO DEBUG MESSAGE


float rangeRandomf(float min, float max) {
    return min + (max-min)*(float(rand())/float(RAND_MAX));
}

int rangeRandomi(int min, int max) {
    return min + rand()%(max-min+1);
}

void  swap(int &a, int &b) {
    int t=a; a = b; b = t;
}

int debegMessage(const char *fmt, ...)
{
#ifndef NDEBUG
    va_list args;
    va_start(args, fmt);
    return printf(fmt, args);
#else
    return 0;
#endif
}

