#include <stdio.h>
#include "StopWatch.h"

////
// Constructor
////
CStopWatch::CStopWatch() : bStarted(false), bPaused(false) {
	initTime();
}

void CStopWatch::initTime() {
#ifdef WIN32
	QueryPerformanceFrequency(&frequency);
	startCount.QuadPart = 0;
	checkCount.QuadPart = 0;
	endCount.QuadPart = 0;
#else
	startCount.tv_sec = startCount.tv_usec = 0;
	checkCount.tv_sec = checkCount.tv_usec = 0;
	endCount.tv_sec = endCount.tv_usec = 0;
#endif
}


void CStopWatch::getCurrentTime(_timeCountType* timeData) {
#ifdef WIN32
	QueryPerformanceCounter(timeData);
#else
	gettimeofday(timeData, NULL);
#endif
}

double CStopWatch::diffTimeInMicroSec(_timeCountType timePre, _timeCountType timeNext) {
#ifdef WIN32
	double startTimeInMicroSec = timePre.QuadPart * (1000000.0 / frequency.QuadPart);
	double endTimeInMicroSec = timeNext.QuadPart * (1000000.0 / frequency.QuadPart);
	return endTimeInMicroSec - startTimeInMicroSec;
#else
	double startTimeInMicroSec = (timePre.tv_sec * 1000000.0) + timePre.tv_usec;
	double endTimeInMicroSec = (timeNext.tv_sec * 1000000.0) + timeNext.tv_usec;
	return endTimeInMicroSec - startTimeInMicroSec;
#endif
}

void CStopWatch::addMicroSeconds(_timeCountType* orgTime, double timeToBeAddedInMicroSec) {
	long addSec = (long)timeToBeAddedInMicroSec / 1000000;
	long addMicro = (long)timeToBeAddedInMicroSec - addSec * 1000000;
#ifdef WIN32
	orgTime->QuadPart += timeToBeAddedInMicroSec * frequency.QuadPart / 1000000;
#else
	orgTime->tv_sec += addSec;
	orgTime->tv_usec += addMicro;
#endif
}




////
// StopWatch starts and records the time
////
void CStopWatch::start() {
	if (bStarted) return;
	else bStarted = true;

	initTime();

	getCurrentTime(&startCount);
	checkCount = startCount;
	endCount = startCount;
}

////
// StopWatch stops and resets the time
////
void CStopWatch::stop() {
	if (!bStarted) return;
	else bStarted = false;
	if (bPaused) { resume(); }

	getCurrentTime(&endCount);

}

// Pause Watch
void CStopWatch::pause() {
	if (!bStarted || bPaused) return;
	else bPaused = true;

	getCurrentTime(&pauseStart);
}
// Resume Watch
void CStopWatch::resume() {
	if (!bPaused) return;
	else bPaused = false;

	getCurrentTime(&pauseEnd);
	double pausedTime = diffTimeInMicroSec(pauseStart, pauseEnd);
	addMicroSeconds(&startCount, pausedTime);
	addMicroSeconds(&checkCount, pausedTime);
}

double CStopWatch::checkAndComputeDT() {
	if (!bStarted || bPaused) return 0.0;

	tempCount = checkCount;
	getCurrentTime(&checkCount);
	return diffTimeInMicroSec(tempCount, checkCount);
}

////
// StopWatch computes the time (in microseconds) between the last "start" and "stop"
////
double CStopWatch::getTotalElapsedTime(){
	if (!bStarted)  return diffTimeInMicroSec(startCount, endCount);
	if (bPaused) {
		endCount = pauseStart;
	}
	else {
		getCurrentTime(&endCount);
	}
	return diffTimeInMicroSec(startCount, endCount);
}