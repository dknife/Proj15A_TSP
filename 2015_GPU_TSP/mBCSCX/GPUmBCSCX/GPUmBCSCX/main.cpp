//
//  main.cpp
//  TSP with mBCSCX_GPU
//
//  Copyright (c) 2015 young-min kang. All rights reserved.
//


#ifdef WIN32 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#endif



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

#define NUMGENES 256
#define NUMGROUPS 2
#define MAXGENERATION 1024000
float MAX_ERROR = 32;


#define COST_XQF131 564
#define COST_XQG237 1019
#define COST_XQL662 2513
#define COST_XQC2175 6830
#define COST_MONALISA 5757191

#define XQF131 "xqf131.txt"
#define XQG237 "xqg237.txt"
#define XQL662 "xql662.txt"
#define XQC2175 "xqc2175.txt"
#define MONALISA "monalisa.txt"

#define NDATA 5
int cost[NDATA] = {
    COST_XQF131, COST_XQG237, COST_XQL662, COST_XQC2175, COST_MONALISA
};
const char* filename[NDATA] {
    XQF131, XQG237, XQL662, XQC2175, MONALISA
};
int currentData = 0;





float evalError(float fit);
void drawCities(void);
void drawPath(int vertList[]) ;
void drawEvolution(void);
void drawSolution(void) ;
void GeneticProcess(void) ;
void idle(void) ;
void display();
void reshape(int w, int h) ;
void keyboard(unsigned char k, int x, int y);
void reset(void) ;
void init(const char *TSPINPUTFILE) ;

CCityLocData cityData;
CGeneticTSPSolver solver;
int curGeneration = 0;
int *bestGene;

int currGeneration = 0;
float err[MAXGENERATION];


bool bSimulate = false;
bool bSimulationOrdered = false;

COpenGLMgr OGLMgr;

float minX, maxX, minY, maxY, maxD;
float offsetY;
float aspRatio;

float evalError(float fit) {
    return (fit-cost[currentData])/cost[currentData];;
}

void drawCities(void) {

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
	for (int i = 0; i<=cityData.numCities; i++) {
		Point loc = cityData.getLocation(vertList[i%cityData.numCities]);
		glVertex3f(loc.x, loc.y, -0.1);
	}
	glEnd();
}

void drawEvolution(void) {

    float sx = (minX+maxD)+offsetY;
    float ex = (minX+maxD)*aspRatio;
    float sy = minY+offsetY;
    float ey = minY+maxD*0.5;
    
    char msg[256];
    sprintf(msg, "Generations: %d", curGeneration);
    OGLMgr.printString(msg, ex-(ex-sx)*0.3, sy-offsetY, 0.0);


    sprintf(msg, "%6.3f", MAX_ERROR);
    OGLMgr.printString(msg, sx-offsetY, ey, 0.0);
    sprintf(msg, "%6.3f", MAX_ERROR/2);
    OGLMgr.printString(msg, sx-offsetY, sy+(ey-sy)*0.5, 0.0);
    
    glLineWidth(1);
    glColor3f(0.0, 0.0, 1.0);

    glBegin(GL_LINES);
    glVertex2d(sx, sy+(ey-sy)*0.5);
    glVertex2d(ex, sy+(ey-sy)*0.5);
    glEnd();
    
    glBegin(GL_LINE_LOOP);
    glVertex2d(sx, sy);
    glVertex2d(ex, sy);
    glVertex2d(ex, ey);
    glVertex2d(sx, ey);
    glEnd();
    
    if(curGeneration<1) return;
    
    glLineWidth(2);
    glColor3f(1.0, 0.0, 0.0);
    float dx = (ex-sx)/curGeneration;
    float x=sx, y=ey;
    glBegin(GL_LINE_STRIP);
    glVertex2d(x, y);
    for(int i=1;i<=curGeneration;i++) {
        x = sx+dx*i;
        y = ( ((err[i]<MAX_ERROR)?err[i]:MAX_ERROR)/MAX_ERROR)*(ey-sy)+sy;
        glVertex2d(x,y);
    }
    glEnd();

}

void drawSolution(void) {

	drawPath(bestGene);
	
}


void GeneticProcess(void) {
    if (bSimulate && curGeneration < MAXGENERATION ) {
        solver.nextGeneration();
        solver.computeFitness();
        solver.copySolution(bestGene);
        curGeneration++;
        err[curGeneration] = evalError(solver.getBestFitness());
    }
    
}

void idle() {
    GeneticProcess();
    glutPostRedisplay();

}


void display() {
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    
   
    glColor3f(0.5, 0.5, 0.5);
    drawSolution();
    glColor3f(1.0, 1.0, 1.0);
    drawCities();
    
    drawEvolution();
    
    
    char msg[256];
    sprintf(msg, "Number of Cities: %d ---  %s", cityData.numCities, bSimulationOrdered?"computing":"stopped");
    float startX = minX+maxD;
    OGLMgr.printString(msg, startX, minY+0.9*maxD, 0.0);
    sprintf(msg, "Number of Genes: %d  | Number of Groups = %d", NUMGENES, NUMGROUPS);
    OGLMgr.printString(msg, startX, minY + 0.8*maxD, 0.0);
    
    
    int bestFitness = solver.getBestFitness();
    sprintf(msg, "BEST GENE FITNESS = %d", bestFitness);
    OGLMgr.printString(msg, startX, minY + 0.7*maxD, 0.0);
    
    sprintf(msg, "(BEST KNOWN PATH: %d) ", cost[currentData]);
    OGLMgr.printString(msg, startX, minY + 0.6*maxD, 0.0);
    sprintf(msg, "   | error (Sol - Best) / Best: %f", evalError(bestFitness));
    OGLMgr.printString(msg, startX, minY + 0.55*maxD, 0.0);
    
    glutSwapBuffers();
    
    bSimulate = bSimulationOrdered;
    
}

void reshape(int w, int h) {
	aspRatio = float(w) / float(h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    offsetY = (maxY - minY)*0.05;
	glOrtho(minX-offsetY*aspRatio, (maxD+minX+offsetY)*aspRatio, minY-offsetY, minY+maxD+offsetY, -1, 1);
}


void keyboard(unsigned char k, int x, int y) {

	switch (k) {
	case 's': bSimulationOrdered = bSimulationOrdered ? false : true; break;
    case 'r': reset(); break;
    case '.': currentData = (currentData+1)%NDATA; reset(); break;
    case ',': currentData = (currentData-1+NDATA)%NDATA; reset(); break;
    case '=': MAX_ERROR /= 2.0; break;
    case '-': MAX_ERROR *= 2.0; break;
    default:
		break;
	}
    glutPostRedisplay();
}


void reset(void) {
    
    if(!bestGene) delete[] bestGene;
    
    init(filename[currentData]);
    curGeneration = 0;
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    offsetY = (maxY - minY)*0.05;
    glOrtho(minX-offsetY*aspRatio, (maxD+minX+offsetY)*aspRatio, minY-offsetY, minY+maxD+offsetY, -1, 1);
    
}

void init(const char *TSPDATAFILE) {

    cityData.readData(TSPDATAFILE);
    minX = cityData.minX;
	minY = cityData.minY;
	maxX = cityData.maxX;
	maxY = cityData.maxY;
    maxD = (maxX-minX)>(maxY-minY)?(maxX-minX):(maxY-minY);

	solver.LoadData(&cityData, NUMGENES, NUMGROUPS);

	bestGene = new int[cityData.numCities];
    
    //if(currentData==NDATA-1) solver.LoadSolution("knownBestTour.txt");
    
    solver.computeFitness();
    solver.copySolution(bestGene);
    err[0] = evalError(solver.getBestFitness()) ;
    
}


int main(int argc, char **argv)
{


	OGLMgr.initGLWindow(&argc, argv, (GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA), 1024, 512, "TSP with GPU mBCSCX");

	init(filename[currentData]);
	

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);

	glutMainLoop();

	return 0;
}

