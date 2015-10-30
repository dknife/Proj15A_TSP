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
#include "GPUTSPSolver.h"

#include "OpenGLMgr.h"

#ifdef WIN32
#define sprintf sprintf_s
#endif

using namespace std;

#define NUMGENES 512
#define NUMGROUPS 12
#define MAXGENERATION 1000000
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
void init(void) ;
void setCamera(void);
void drawGene(void);

CCityLocData cityData;
CGPUTSPSolver solver;
int curGeneration = 0;
int *bestGene;

int currGeneration = 0;
float err[MAXGENERATION];
bool  record[MAXGENERATION];


bool bSimulate = false;
bool bSimulationOrdered = false;
bool bGeneView = false;

COpenGLMgr OGLMgr;

float minX, maxX, minY, maxY, maxD;
float offsetY;
float aspRatio;
int width = 0;
int height = 0;
float geneDrawScaleX = 1.0;
float geneDrawScaleY = 1.0;

float evalError(float fit) {
    return (fit-cost[currentData])/cost[currentData];;
}

void drawCities(void) {

    glPushMatrix();
    glTranslatef(-offsetY, minY+maxD-(maxY-minY), 0.0);
    glPointSize(2);
	glBegin(GL_POINTS);
	for (int i = 0; i<cityData.numCities; i++) {
		Point loc = cityData.getLocation(i);
		glVertex2f(loc.x, loc.y);
	}
	glEnd();
    glPopMatrix();
}

void drawPath(int vertList[]) {

    glLineWidth(1);
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


    sprintf(msg, "%6.2f", MAX_ERROR);
    OGLMgr.printString(msg, sx-offsetY*2, ey, 0.0);
    sprintf(msg, "%6.2f", MAX_ERROR/2);
    OGLMgr.printString(msg, sx-offsetY*2, sy+(ey-sy)*0.5, 0.0);
    sprintf(msg, "0.0", MAX_ERROR/2);
    OGLMgr.printString(msg, sx-offsetY, sy, 0.0);
    
    sprintf(msg, "%d cities, %d genes (%d groups)", solver.getNumCities(), NUMGENES, NUMGROUPS);
    OGLMgr.printString(msg, sx+(ex-sx)/2, ey-offsetY, 0.0);
    sprintf(msg, "crossover: %s", solver.getCrossoverMethod(), NUMGENES);
    OGLMgr.printString(msg, sx+(ex-sx)/2, ey-offsetY*2, 0.0);
    if(solver.bHeating) sprintf(msg, "Heating On - cycle: %d generation", solver.nCycleGeneration, NUMGENES);
    else sprintf(msg, "Heating Off");
    OGLMgr.printString(msg, sx+(ex-sx)/2, ey-offsetY*3, 0.0);
    sprintf(msg, "Best Gene (error: %4.2f %%)", 100.0*evalError(solver.getFitRecord()));
    OGLMgr.printString(msg, sx+(ex-sx)/2, ey-offsetY*4, 0.0);
    
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

    glLineWidth(1);
    
    float bestLocx = sx;
    float bestLocy = sy;
    glColor3f(0.8, 0.8, 1);
    glBegin(GL_LINES);
    for(int i=1;i<=curGeneration;i++) {
        if(record[i]) {
            x = sx+dx*i;
            y = ( ((err[i]<MAX_ERROR)?err[i]:MAX_ERROR)/MAX_ERROR)*(ey-sy)+sy;
            glVertex2d(x,sy);
            glVertex2d(x,y);
            bestLocx = x;
            bestLocy = y;
        }
    }
    glVertex2d(bestLocx, sy);
    glVertex2d(bestLocx, bestLocy);
    glEnd();

}


void drawSolution(void) {

    glLineWidth(2);
    glDisable(GL_DEPTH_TEST);
    glPushMatrix();
    glTranslatef(-offsetY, minY+maxD-(maxY-minY), 0.0);
    solver.copySolution(bestGene);
	drawPath(bestGene);
    glPopMatrix();

    glLineWidth(1);
    glPushMatrix();
    glTranslatef(maxX, minY + 0.75*maxD, 0.0);
    glScalef(0.3, 0.3, 0.3);
    glColor3f(0.0, 0.0, 1.0);
    solver.copyRecordHolder(bestGene);
    glLineWidth(1);
    drawPath(bestGene);
    glPopMatrix();
    
    char msg[256];
    sprintf(msg, "Best Gene (error: %4.2f %%)", 100.0*evalError(solver.getFitRecord()));
    OGLMgr.printString(msg, maxX + maxX*0.3, minY + 0.75*maxD, 0.0);

	
}

void drawGene(void) {
    
    if(!bGeneView) return;
    
    float dx = 60, dy = 20;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    glOrtho(0, width, 0, height, -10, 10);
    int nCities = solver.getNumCities();
    
    geneDrawScaleX = width/(2.5*nCities);
    
    solver.drawGene(dx, dy, geneDrawScaleX, geneDrawScaleY);
    
    int nG = NUMGENES/NUMGROUPS;
    char msg[256];
    for (int i=0; i<NUMGROUPS; i++) {
        sprintf(msg, "group %d", i);
        OGLMgr.printString(msg, dx + geneDrawScaleX * solver.getNumCities() , dy + geneDrawScaleY*i*nG, 0);
    }
    
    sprintf(msg, "best_");
    OGLMgr.printString(msg, 0 , dy + geneDrawScaleY*solver.getBestGeneIdx(), 0);
}




void GeneticProcess(void) {
    if (bSimulate && curGeneration < MAXGENERATION ) {
        
        solver.nextGeneration();
        
        solver.computeFitness();
        curGeneration++;
        err[curGeneration] = evalError(solver.getBestFitness());
        record[curGeneration] = solver.recordBroken;
        
        //solver.fixGene(solver.getBestGeneIdx());
        

    }
    
}

void idle() {
    GeneticProcess();
    glutPostRedisplay();

}


void display() {
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    setCamera();
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    
   
    glColor3f(0.5, 0.5, 0.5);
    drawSolution();
    
    glColor3f(0.0, 0.0, 0.0);
    drawCities();
    
    drawEvolution();
    
    
    char msg[256];
    sprintf(msg, "Number of Cities: %d ---  %s (Temp: %4.1f) ", cityData.numCities, bSimulationOrdered?"computing":"stopped", solver.getTemerature());
    float startX = minX+maxD;
    OGLMgr.printString(msg, startX, minY + 0.6*maxD, 0.0);
    
    

    int bestFit = solver.getBestFitness();
    sprintf(msg, " | Known Optimal: %d / Best Gene: %d / Record: %d", cost[currentData], bestFit, solver.getFitRecord());
    OGLMgr.printString(msg, startX, minY + 0.57*maxD, 0.0);
    sprintf(msg, " | error : %f (record=%f)", evalError(bestFit), evalError(solver.getFitRecord()));
    OGLMgr.printString(msg, startX, minY + 0.55*maxD, 0.0);
    
    
    drawGene();
    
    glutSwapBuffers();
    
    bSimulate = bSimulationOrdered;
    
}

void setCamera(void) {
    glViewport(0, 0, width,height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    offsetY = (maxY - minY)*0.05;
    glOrtho(minX-offsetY*aspRatio, (maxD+minX+offsetY)*aspRatio, minY-offsetY, minY+maxD+offsetY, -1, 1);
}

void reshape(int w, int h) {
    width = w;
    height = h;
	aspRatio = float(w) / float(h);
    setCamera();
}


void keyboard(unsigned char k, int x, int y) {

	switch (k) {
	case 's': bSimulationOrdered = bSimulationOrdered ? false : true; break;
    case 'r': reset(); break;
    case '.': currentData = (currentData+1)%NDATA; reset(); break;
    case ',': currentData = (currentData-1+NDATA)%NDATA; reset(); break;
    case '=': MAX_ERROR /= 2.0; break;
    case '-': MAX_ERROR *= 2.0; break;
    case 'a': geneDrawScaleY *= 1.05; break;
    case 'z': geneDrawScaleY *= 0.95; break;
    case 'i': bGeneView = bGeneView?false:true; break;
    case 'x': solver.changeCrossoverMethod(); break;
        case 'h': solver.bHeating = solver.bHeating?false:true; break;
    default:
		break;
	}
    glutPostRedisplay();
}

void setupCamera(void) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	offsetY = (maxY - minY)*0.05;
	glOrtho(minX - offsetY*aspRatio, (maxD + minX + offsetY)*aspRatio, minY - offsetY, minY + maxD + offsetY, -1, 1);
}

void reset(void) {
    
    if(!bestGene) delete[] bestGene;
    
	cityData.readData(filename[currentData]);

	minX = cityData.minX;
	minY = cityData.minY;
	maxX = cityData.maxX;
	maxY = cityData.maxY;
	maxD = (maxX - minX)>(maxY - minY) ? (maxX - minX) : (maxY - minY);

	solver.LoadData(&cityData, NUMGENES, NUMGROUPS);
	solver.initSolver();

	bestGene = new int[cityData.numCities];
	solver.computeFitness();
	solver.copySolution(bestGene);
	err[0] = evalError(solver.getBestFitness());

    curGeneration = 0;
    


	setupCamera();
    
    
}

void init(void) {

	reset();

}


int main(int argc, char **argv)
{

	init();

	OGLMgr.initGLWindow(&argc, argv, (GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA), 1024, 512, "TSP with GPU mBCSCX");

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);



	glutMainLoop();

	return 0;
}

