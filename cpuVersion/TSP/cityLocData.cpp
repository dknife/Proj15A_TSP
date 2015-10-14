//
//  cityLocData.cpp
//  for Mona Lisa TSP
//
//  Copyright (c) 2015 young-min kang. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include "utilities.h"
#include <math.h>

#include "cityLocData.h"
using namespace std;


CCityLocData::CCityLocData(int nCities) : numCities(nCities) {

	cityLoc = new Point[numCities];
	for (int i = 0; i<numCities; i++) {
		cityLoc[i].x = rangeRandomf(0.0, 0.4)* ((rand() % 2) ? 1 : -1);
		cityLoc[i].y = rangeRandomf(0.0, 0.4)* ((rand() % 2) ? 1 : -1);
		if (i == 0) {
			minX = maxX = cityLoc[i].x;
			minY = maxY = cityLoc[i].y;
		}
		else {
			if (cityLoc[i].x < minX) minX = cityLoc[i].x;
			if (cityLoc[i].x > maxX) maxX = cityLoc[i].x;
			if (cityLoc[i].y < minY) minY = cityLoc[i].y;
			if (cityLoc[i].y > maxY) maxY = cityLoc[i].y;
		}
	}
}

float CCityLocData::cityDistance(int i, int j) {
	float dx = cityLoc[i].x - cityLoc[j].x;
	float dy = cityLoc[i].y - cityLoc[j].y;
	return sqrt(dx*dx + dy*dy);
}


Point CCityLocData::getLocation(int cityIdx) {
	return cityLoc[cityIdx];
}

void CCityLocData::setLocation(int cityIdx, Point p) {
	cityLoc[cityIdx].x = p.x;
	cityLoc[cityIdx].y = p.y;
}