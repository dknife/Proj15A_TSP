//
//  main.cpp
//  TSP
//
//  Created by young-min kang on 7/25/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//



#include <iostream>
#include <iomanip>
#include <math.h>

#include "cityLocData.h"
#include "GeneticTSPSolver.h"
#include "OpenGLMgr.h"
#include "StopWatch.h"

#ifdef WIN32
#define sprintf sprintf_s
#endif

using namespace std;

bool bSimulate = false;
bool bSCX = false;
bool bTWSCX = false;
bool bBCSCX = false;
bool bmBCSCX = false;

#define SCX_COLOR           0.0, 0.0, 0.0
#define TWSCX_COLOR         0.0, 0.0, 1.0
#define BCSCX_COLOR         0.0, 0.5, 0.0
#define BCSCX_MULTI_COLOR   1.0, 0.0, 0.0

GLfloat fogColor[] = { 0.0, 0.0, 0.0 };

#define NUMCITIES 2000 //4056
#define NUMGENES  1024
#define NUMGENERATIONS 1000
#define NUM_POP_GROUPS 4

CCityLocData cityData(NUMCITIES);

CGeneticTSPSolver tspSolverPheno_SCX(&cityData, NUMGENES, 1);
CGeneticTSPSolver tspSolverPheno_TWSCX(&cityData, NUMGENES, 1);
CGeneticTSPSolver tspSolverPheno_BCSCX(&cityData, NUMGENES, 1);
CGeneticTSPSolver tspSolverMulti_BCSCX(&cityData, NUMGENES, NUM_POP_GROUPS);
int curGeneration = 0;

CStopWatch myWatch;

COpenGLMgr OGLMgr;

int     curSolutionPheno_SCX[NUMCITIES];
int     curSolutionPheno_BCSCX[NUMCITIES];
int     curSolutionPheno_TWSCX[NUMCITIES];
int     curSolutionMulti_BCSCX[NUMCITIES];

float   fitnessGraphPheno_SCX[NUMGENERATIONS];
float   fitnessGraphPheno_BCSCX[NUMGENERATIONS];
float   fitnessGraphPheno_TWSCX[NUMGENERATIONS];
float   fitnessGraphMulti_BCSCX[NUMGENERATIONS];

void drawCities(void) {
    
    glColor3f(0.25, 0.25, 0.25);
    glPointSize(3);
    
    glBegin(GL_POINTS);
    for(int i=0;i<NUMCITIES;i++) {
        Point loc = cityData.getLocation(i);
        glVertex2f(loc.x, loc.y);
    }
    glEnd();
    
}

void drawPath(int vertList[]) {
    
    glLineWidth(2);
    glBegin(GL_LINE_STRIP);
    for(int i=0;i<NUMCITIES;i++) {
		Point loc = cityData.getLocation(vertList[i]);
        glVertex2f(loc.x, loc.y);
    }
    glEnd();
    
    

}
void drawSolution(void) {
    
    float fitSCX    = tspSolverPheno_SCX.getBestFitness();
    float fitTWSCX = tspSolverPheno_TWSCX.getBestFitness();
    float fitBCSCX  = tspSolverPheno_BCSCX.getBestFitness();
    float fitBCSCX2  = tspSolverMulti_BCSCX.getBestFitness();
    
    char strSCX[64], strTWSCX[64], strBCSCX[64], strBCSCX2[64];
    sprintf(strSCX, "  SCX (d= %5.2f)", fitSCX);
    sprintf(strTWSCX, " TWSCX (d= %5.2f)", fitTWSCX);
    sprintf(strBCSCX, "BCSCX (d= %5.2f)", fitBCSCX);
    sprintf(strBCSCX2, "mBCSCX (d= %5.2f)", fitBCSCX2);
    glPushMatrix();
    glTranslatef(-0.5, 0.5, 0.0);
    drawCities();
    glColor3f(SCX_COLOR);
    drawPath(curSolutionPheno_SCX);
    OGLMgr.printString(strSCX, -0.25, -0.475, 0);
    glPopMatrix();
    
    glPushMatrix();
    glTranslatef( 0.5, 0.5, 0.0);
    drawCities();
    glColor3f(TWSCX_COLOR);
    drawPath(curSolutionPheno_TWSCX);
    OGLMgr.printString(strTWSCX, -0.25, -0.475, 0);
    glPopMatrix();
    
    glPushMatrix();
    glTranslatef(-0.5, -0.5, 0.0);
    drawCities();
    glColor3f(BCSCX_COLOR);
    drawPath(curSolutionPheno_BCSCX);
    OGLMgr.printString(strBCSCX, -0.25, 0.425, 0);
    glPopMatrix();
    
    glPushMatrix();
    glTranslatef(0.5, -0.5, 0.0);
    drawCities();
    glColor3f(BCSCX_MULTI_COLOR);
    drawPath(curSolutionMulti_BCSCX);
    OGLMgr.printString(strBCSCX2, -0.25, 0.425, 0);
    glPopMatrix();
    
    glBegin(GL_LINES);
    glVertex2f(-1.0,  0.0);
    glVertex2f( 1.0,  0.0);
    glVertex2f( 0.0, -1.0);
    glVertex2f( 0.0,  1.0);
    glEnd();

    
    
}

void drawGraph(int nGen) {
    
    OGLMgr.printString("Best Solutions of Each Generation", 0.4, 1.1, 0.0);
    
    char cityInfo[64];
    char populationInfo[64];
    char multiPopInfo[64];
    char fitHigh[64];
    char fitMiddle[64];
    char genMiddle[64];
    char genLast[64];
    
    glColor4f(0.8, 0.8, 0.8, 1.0);
    glBegin(GL_QUADS);
    glVertex3f(0.0, -1.0, -1.0);
    glVertex3f(2.0, -1.0, -1.0);
    glVertex3f(2.0,  1.0, -1.0);
    glVertex3f(0.0,  1.0, -1.0);
    glEnd();
    
    sprintf(cityInfo,       "Num. of Cities     = %d", NUMCITIES);
    sprintf(populationInfo, "Population Size    = %d", NUMGENES);
    sprintf(multiPopInfo,   "Num. of Pop. Groups for BCSCX-multi = %d", NUM_POP_GROUPS);
    OGLMgr.printString(cityInfo, 0.20, 0.9, 0.0);
    OGLMgr.printString(populationInfo, 0.20, 0.8, 0.0);
    OGLMgr.printString(multiPopInfo, 0.20, 0.7, 0.0);
    
    sprintf(fitHigh,   "%5.2f", NUMCITIES/2.0);
    sprintf(fitMiddle, "%5.2f", NUMCITIES/4.0);
    sprintf(genLast,   "%d", nGen-1);
    sprintf(genMiddle, "%.1f", (nGen-1)/2.0);
    
    OGLMgr.printString("Distance", -0.2, 1.02, 0.0);
    OGLMgr.printString(fitHigh,  -0.2, 0.95, 0.0);
    OGLMgr.printString(fitMiddle,-0.2,-0.05, 0.0);
    OGLMgr.printString("0",-0.1,-1.0, 0.0);
    
    OGLMgr.printString("Generation", 1.0, -1.2, 0.0);
    OGLMgr.printString("0", 0.0,-1.1, 0.0);
    OGLMgr.printString(genMiddle,  1.0, -1.1, 0.0);
    OGLMgr.printString(genLast,2.0, -1.1, 0.0);

    
    
    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);
    // fitness lines
    glVertex2f( 0.0, -1.0);
    glVertex2f(-0.05, -1.0);
    glVertex2f( 0.0,  0.0);
    glVertex2f(-0.05,  0.0);
    glVertex2f( 0.0,  1.0);
    glVertex2f(-0.05,  1.0);
    // number of generations lines
    glVertex2f( 0.0, -1.0);
    glVertex2f( 0.0, -1.05);
    glVertex2f( 1.0, -1.0);
    glVertex2f( 1.0, -1.05);
    glVertex2f( 2.0, -1.0);
    glVertex2f( 2.0, -1.05);
    glEnd();
    
    glLineWidth(2);
    glColor3f(SCX_COLOR);
    glBegin(GL_LINE_STRIP);
    for (int i=0; i<NUMGENERATIONS && i<nGen; i++) {
        glVertex3f(float(i*2.0)/(nGen-1),4.0*fitnessGraphPheno_SCX[i]/NUMCITIES - 1.0,0);
    }
    glEnd();
    
    glColor3f(TWSCX_COLOR);
    glBegin(GL_LINE_STRIP);
    for (int i=0; i<NUMGENERATIONS && i<nGen; i++) {
        glVertex3f(float(i*2.0)/(nGen-1),4.0*fitnessGraphPheno_TWSCX[i]/NUMCITIES - 1.0,0);
    }
    glEnd();
    
    
    glColor3f(BCSCX_COLOR);
    glBegin(GL_LINE_STRIP);
    for (int i=0; i<NUMGENERATIONS && i<nGen; i++) {
        glVertex3f(float(i*2.0)/(nGen-1),4.0*fitnessGraphPheno_BCSCX[i]/NUMCITIES - 1.0,0);
    }
    glEnd();
    
    glColor3f(BCSCX_MULTI_COLOR);
    glBegin(GL_LINE_STRIP);
    for (int i=0; i<NUMGENERATIONS && i<nGen; i++) {
        glVertex3f(float(i*2.0)/(nGen-1),4.0*fitnessGraphMulti_BCSCX[i]/NUMCITIES - 1.0,0);
    }
    glEnd();
    glLineWidth(1);
}
void display() {
	
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    OGLMgr.placeCamera(0.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    
    
    if(curGeneration<NUMGENERATIONS && bSimulate) {
        
        if(curGeneration<NUMGENERATIONS) fitnessGraphPheno_TWSCX[curGeneration] = tspSolverPheno_TWSCX.getBestFitness();
        if(curGeneration<NUMGENERATIONS) fitnessGraphMulti_BCSCX[curGeneration] = tspSolverMulti_BCSCX.getBestFitness();
        if(curGeneration<NUMGENERATIONS) fitnessGraphPheno_SCX[curGeneration]   = tspSolverPheno_SCX.getBestFitness();
        if(curGeneration<NUMGENERATIONS) fitnessGraphPheno_BCSCX[curGeneration] = tspSolverPheno_BCSCX.getBestFitness();
        
        
        double timeSCX, timeBCSCX, timeTWSCX,timemBCSCX;
        myWatch.checkAndComputeDT();
        if(bSCX) tspSolverPheno_SCX.nextGeneration(PHENOTYPE_SCX);
        timeSCX = myWatch.checkAndComputeDT();
        if(bTWSCX) tspSolverPheno_TWSCX.nextGeneration(PHENOTYPE_TWSCX);
        timeTWSCX = myWatch.checkAndComputeDT();
        if(bBCSCX) tspSolverPheno_BCSCX.nextGeneration(PHENOTYPE_BCSCX);
        timeBCSCX = myWatch.checkAndComputeDT();
        if(bmBCSCX) tspSolverMulti_BCSCX.nextGeneration(PHENOTYPE_BCSCX);
        timemBCSCX = myWatch.checkAndComputeDT();
		printf("time = %lf %lf %lf %lf\n", timeSCX / NUMGENES, timeTWSCX / NUMGENES, timeBCSCX / NUMGENES, timemBCSCX / NUMGENES);
        
        if(bSCX) tspSolverPheno_SCX.computeFitness();
        if(bTWSCX) tspSolverPheno_TWSCX.computeFitness();
        if(bBCSCX) tspSolverPheno_BCSCX.computeFitness();
        if(bmBCSCX) tspSolverMulti_BCSCX.computeFitness();
        
        if(bSCX) tspSolverPheno_SCX.copySolution(curSolutionPheno_SCX);
        if(bTWSCX) tspSolverPheno_TWSCX.copySolution(curSolutionPheno_TWSCX);
        if(bBCSCX) tspSolverPheno_BCSCX.copySolution(curSolutionPheno_BCSCX);
        if(bmBCSCX) tspSolverMulti_BCSCX.copySolution(curSolutionMulti_BCSCX);
        
        curGeneration++;
        
	}
    
    glPushMatrix();
    glTranslatef(0.2, 0.0, 0.0);
    glScalef(0.8, 0.8, 0.8);
    drawGraph(curGeneration);
    glPopMatrix();
	   
    glPushMatrix();
    glTranslatef(-1.0, 0.0, 0.0);
    drawSolution();
    glPopMatrix();
    
	glutSwapBuffers();
	
}

void reshape(int w, int h) {
	float aspRatio = float(w)/float(h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	OGLMgr.initOrthoCamera(-2, 2, -1, 1, -1, 1);
}

void keyboard(unsigned char k, int x, int y) {
    
    switch (k) {
        case 's':
            bSimulate = (bSimulate)?false:true;
            break;
        case '1':
            bSCX = bSCX?false:true; break;
        case '2':
            bTWSCX = bTWSCX?false:true; break;
        case '3':
            bBCSCX = bBCSCX?false:true; break;
        case '4':
            bmBCSCX = bmBCSCX?false:true; break;
        case 'r':
            tspSolverPheno_TWSCX.initSolver();
            tspSolverPheno_SCX.initSolver();
            tspSolverPheno_BCSCX.initSolver();
            tspSolverMulti_BCSCX.initSolver();
            tspSolverPheno_TWSCX.computeFitness();
            tspSolverPheno_SCX.computeFitness();
            tspSolverPheno_BCSCX.computeFitness();
            tspSolverMulti_BCSCX.computeFitness();
            curGeneration = 0;
            break;
        default:
            break;
    }
}
int main(int argc, char **argv)
{
    
    
    OGLMgr.initGLWindow(&argc, argv, (GLUT_DOUBLE|GLUT_DEPTH|GLUT_RGBA), 1024,512, "Travelling Salesman Problem - Evolutionary Approach");
    OGLMgr.initOrthoCamera(-2, 2, -1, 1, -1, 1);
    
    //mat.printMatrix();
    
    tspSolverPheno_TWSCX.computeFitness();
    tspSolverPheno_SCX.computeFitness();
    tspSolverPheno_BCSCX.computeFitness();
    tspSolverMulti_BCSCX.computeFitness();
    
    myWatch.start();

    
    glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    
	glutMainLoop();
    
    return 0;
}

