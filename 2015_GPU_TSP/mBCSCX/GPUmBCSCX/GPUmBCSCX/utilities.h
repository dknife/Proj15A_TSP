//
//  utilities.h
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#ifndef __TSP__utilities__
#define __TSP__utilities__

#include <iostream>


#ifdef WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif


float rangeRandomf(float min, float max);
int   rangeRandomi(int min, int max);
void  swap(int &a, int &b);


int   debegMessage(const char *fmt, ...);

#endif /* defined(__TSP__utilities__) */
