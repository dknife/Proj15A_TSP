//
//  main.cpp
//  TSP with mBCSCX_GPU
//
//  Copyright (c) 2015 young-min kang. All rights reserved.
//


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include <stdio.h>



#include <iostream>
#include <iomanip>
#include <math.h>

#include "cityLocData.h"
#include "GeneticTSPSolver.h"
#include "OpenGLMgr.h"

#ifdef WIN32
#define sprintf sprintf_s
#endif

using namespace std;

#define NUMGENES 100
#define NUMGROUPS 3

CCityLocData cityData;
CGeneticTSPSolver solver;
int curGeneration = 0;
int *bestGene;

bool bSimulate = false;

COpenGLMgr OGLMgr;

float minX, maxX, minY, maxY;
float aspRatio;

void drawCities(void) {

	glColor3f(0.25, 0.25, 0.25);

	glBegin(GL_POINTS);
	for (int i = 0; i<cityData.numCities; i++) {
		Point loc = cityData.getLocation(i);
		glVertex2f(loc.x, loc.y);
	}
	glEnd();
}

void drawPath(int vertList[]) {

	glLineWidth(2);
	glBegin(GL_LINE_STRIP);
	for (int i = 0; i<cityData.numCities; i++) {
		Point loc = cityData.getLocation(vertList[i]);
		glVertex3f(loc.x, loc.y, -0.1);
	}
	glEnd();



}
void drawSolution(void) {

	drawPath(bestGene);
	
}



void display() {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity(); 
	

	if (bSimulate) {
		solver.nextGeneration();
		solver.computeFitness();
		solver.copySolution(bestGene);
	}

	glColor3f(0.5, 0.5, 0.5);
	drawSolution();
	glColor3f(0.0, 0.0, 0.0);
	drawCities();

	

	char msg[256];
	sprintf(msg, "Number of Cities: %d", cityData.numCities);
	OGLMgr.printString(msg, maxX, minY+0.9*(maxY-minY), 0.0);
	
	sprintf(msg, "Number of Genes: %d  | Number of Groups = %d", NUMGENES, NUMGROUPS);
	OGLMgr.printString(msg, maxX, minY + 0.8*(maxY - minY), 0.0);

	float bestFitness = solver.getBestFitness();
	sprintf(msg, "BEST GENE FITNESS = %f", bestFitness);
	OGLMgr.printString(msg, maxX, minY + 0.7*(maxY - minY), 0.0);

	sprintf(msg, "(BEST KNOWN PATH: 5,757,191) ");
	OGLMgr.printString(msg, maxX, minY + 0.6*(maxY - minY), 0.0);
	sprintf(msg, "    | RATE (our solution/known-best): %f", bestFitness / 5757191);
	OGLMgr.printString(msg, maxX, minY + 0.55*(maxY - minY), 0.0);



	glutSwapBuffers();

}

void reshape(int w, int h) {
	aspRatio = float(w) / float(h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(minX, maxX*aspRatio, minY, maxY, -1, 1);
}

void keyboard(unsigned char k, int x, int y) {

	switch (k) {
	case 27: exit(0);
	case 's': bSimulate = bSimulate ? false : true; break;
	default:
		break;
	}
}

void init(void) {

	cityData.readData("monalisa.txt");
	minX = cityData.minX;
	minY = cityData.minY;
	maxX = cityData.maxX;
	maxY = cityData.maxY;

	solver.LoadData(&cityData, NUMGENES, NUMGROUPS);

	bestGene = new int[cityData.numCities];
	for (int i = 0; i < cityData.numCities; i++) {
		bestGene[i] = i;
	}
}


int main(int argc, char **argv)
{


	OGLMgr.initGLWindow(&argc, argv, (GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA), 1024, 512, "TSP with GPU mBCSCX");

	init();
	

	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);

	glutMainLoop();

	return 0;
}

