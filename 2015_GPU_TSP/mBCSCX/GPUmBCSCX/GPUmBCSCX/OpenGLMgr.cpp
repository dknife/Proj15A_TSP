//
//  OpenGLMgr.cpp
//  TSP
//
//  Created by young-min kang on 8/4/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#include "OpenGLMgr.h"

COpenGLMgr::COpenGLMgr() {
    
}
    
void COpenGLMgr::initGLWindow(int *pArgc, char **ppArgv, unsigned int bufferMode, GLint width, GLint height, char *title) {
    glutInit(pArgc, ppArgv);
    glutInitDisplayMode(bufferMode);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(width,height);
	glutCreateWindow(title);
    
    if(GLUT_DOUBLE & bufferMode) glEnable(GL_DEPTH_TEST);
    

	glClearColor(1.0, 1.0, 1.0, 1.0);
}

void COpenGLMgr::initPerspectiveCamera(GLdouble fovy, GLdouble aspRatio, GLdouble nearPlane, GLdouble farPlane) {
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, aspRatio, nearPlane, farPlane);
}

void COpenGLMgr::initOrthoCamera(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble nearPlane, GLdouble farPlane) {
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(left, right, bottom, top, nearPlane, farPlane);
}


void COpenGLMgr::printString(const char *str, int x, int y, float w, float h, float color[4])
{
	
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
    glDisable(GL_LIGHTING);     // need to disable lighting for proper text color
    glDisable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	
	
	
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, w, h, 0, -1.0, 1.0);
	
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	
	
	
	
    if(color) glColor4fv(color);          // set text color
	else glColor4f(0.0,0.0,0.0,1.0);
	
    glRasterPos2i(x, y);        // place text position
	
    // loop all characters in the string
    while(*str)
    {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *str);
        ++str;
    }
	
	glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    
	
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	
	glMatrixMode(GL_MODELVIEW);
	glPopAttrib();
}

void COpenGLMgr::printString(const char *str, float x, float y, float z, float color[4])
{
    glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
	glDisable(GL_LIGHTING);     // need to disable lighting for proper text color
    glDisable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	
    if(color) glColor4fv(color);          // set text color
    else glColor4f(0.0, 0.0, 0.0, 1.0);
    
    glRasterPos3f(x,y,z);        // place text position
	
	// loop all characters in the string
    while(*str)
    {
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *str);
        ++str;
    }
	/*
     #define GLUT_BITMAP_9_BY_15		(&glutBitmap9By15)
     #define GLUT_BITMAP_8_BY_13		(&glutBitmap8By13)
     #define GLUT_BITMAP_TIMES_ROMAN_10	(&glutBitmapTimesRoman10)
     #define GLUT_BITMAP_TIMES_ROMAN_24	(&glutBitmapTimesRoman24)
     #if (GLUT_API_VERSION >= 3)
     #define GLUT_BITMAP_HELVETICA_10	(&glutBitmapHelvetica10)
     #define GLUT_BITMAP_HELVETICA_12	(&glutBitmapHelvetica12)
     #define GLUT_BITMAP_HELVETICA_18	(&glutBitmapHelvetica18)
     */
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glPopAttrib();
}