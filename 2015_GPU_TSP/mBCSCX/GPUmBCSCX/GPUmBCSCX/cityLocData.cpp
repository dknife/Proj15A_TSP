//
//  cityLocData.cpp
//  for Mona Lisa TSP
//
//  Copyright (c) 2015 young-min kang. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"
#include <math.h>

#ifdef WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include "cityLocData.h"
using namespace std;


CCityLocData::CCityLocData() : numCities(0) {

	cityLoc = NULL;
	minX = maxX = minY = maxY = 0.0;
	
}



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


void CCityLocData::readData(const char *fName) {

	int cityId;
	FILE *fInput = fopen(fName, "r");
	if (fInput == NULL) {
		printf("file not found : %s\n", fName);
		char pathStr[256];
		GetCurrentDir(pathStr, sizeof(pathStr));
		printf("working dir: %s\n", pathStr);

		exit(1);
	}

	printf("file loading started...\n");

	fscanf(fInput, "%d\n", &numCities);
	printf("nCities: %d\n", numCities);
	cityLoc = new Point[numCities];
	for (int i = 0; i<numCities; i++) {
		fscanf(fInput, "%d", &cityId);
		fscanf(fInput, "%f", &cityLoc[i].x);
		fscanf(fInput, "%f", &cityLoc[i].y);
		if (cityId - 1 != i) {
			printf("Data Invalid\n"); exit(1);
		}
		
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
	printf("file successfully loaded!\n");
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