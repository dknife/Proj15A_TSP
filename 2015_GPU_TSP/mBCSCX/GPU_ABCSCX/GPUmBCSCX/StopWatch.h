#ifndef _STOPWATCH_YMKANG_H
#define _STOPWATCH_YMKANG_H

////////////////////////////////
//
// StopWatch for Measuring Time
//
// Young-Min Kang
// Tongmyong University

#ifdef WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif

#ifdef WIN32
typedef LARGE_INTEGER _timeCountType;
#else
typedef timeval _timeCountType;
#endif

class CStopWatch {
	bool bStarted;
	bool bPaused;

#ifdef WIN32
	_timeCountType frequency; // ticks per seconds (required only on Windows)
#endif
	_timeCountType startCount;
	_timeCountType endCount;
	_timeCountType checkCount;
	_timeCountType tempCount;
	_timeCountType pauseStart;
	_timeCountType pauseEnd;

	void    getCurrentTime(_timeCountType* timeData);
	double  diffTimeInMicroSec(_timeCountType  timePre, _timeCountType timeNext);
	void    addMicroSeconds(_timeCountType* orgTime, double timeToBeAddedInMicroSec);

	void initTime();

	// public methods
public:
	CStopWatch();
	void start();       // start  StopWatch : watch start at time=0
	void stop();        // stops  StopWatch : watch stops and holds the end-time
	void pause();       // pauses StopWatch : time does not pass when the watch paused
	void resume();      // resume StopWatch : resume time to pass again from the paused moment
	bool bRunning() { return bStarted; }

	// time checking ( in microseconds)
	double checkAndComputeDT();     // check time and returns the "delta time" since the previous time check
	double getTotalElapsedTime();   // return the total elapsed time since the StopWatch started
};

#endif