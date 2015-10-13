//
//  OpenGLMgr.h
//  TSP
//
//  Created by young-min kang on 8/4/14.
//  Copyright (c) 2014 young-min kang. All rights reserved.
//

#ifndef __TSP__OpenGLMgr__
#define __TSP__OpenGLMgr__

#ifdef WIN32
#include <windows.h>
//#include <gl/glew.h>
#include <gl/gl.h>
#include <gl/glut.h>
#else // MAC OS X
#include <OpenGL/OpenGL.h>
#include <GLUT/GLUT.h> // OpenGL utility toolkit
#endif

#include <iostream>

class COpenGLMgr {
public:
    COpenGLMgr();

    void initGLWindow(int *pArgc, char **ppArgv, unsigned int bufferMode, GLint width=512, GLint height=512, char *title=NULL);
    void initPerspectiveCamera(GLdouble fovy, GLdouble aspRatio, GLdouble nearPlane, GLdouble farPlane);
    void initOrthoCamera(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble nearPlane, GLdouble farPlane);
    void placeCamera(GLdouble x, GLdouble y, GLdouble z, GLdouble tx, GLdouble ty, GLdouble tz, GLdouble ux=0, GLdouble uy=1, GLdouble uz=0);
    
    void setFog(GLfloat density, GLfloat *fogColor);
    void printString(const char *str, int x, int y, float w, float h, float color[4] = NULL);
    void printString(const char *str, float x, float y, float z, float color[4]=NULL);
    
};
#endif /* defined(__TSP__OpenGLMgr__) */

