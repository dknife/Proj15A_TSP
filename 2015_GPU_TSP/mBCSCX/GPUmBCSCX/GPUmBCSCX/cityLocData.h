//
//  cityLocData.h
//  for Mona Lisa TSP Project
//  Copyright (c) 2015 young-min kang. All rights reserved.
//

#ifndef __TSP__CITYLOCDATA_HH__
#define __TSP__CITYLOCDATA_HH__

struct Point {
	float x;
	float y;
};

class CCityLocData {
	Point *cityLoc;

public:
	int numCities;
	// bounding rectangle
	float minX, maxX;
	float minY, maxY;

	// methods
public:
	CCityLocData();
	CCityLocData(int nCities);


	void  readData(const char *filename);
	Point getLocation(int cityIdx);
	void  setLocation(int cityIdx, Point p);
	float cityDistance(int i, int j);
};

#endif /* defined(__TSP__CITYLOCDATA_HH__) */
