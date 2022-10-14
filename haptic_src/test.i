%module test


%{
#define SWIG_FILE_WITH_INIT
#include "INCLUDE/test.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* retrived_info, int n1)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* retrived_info2, int n2)}

int initTouch_right();
int initTouch_left();
void startScheduler();
void stopScheduler();
void closeTouch_left();
void closeTouch_right();
void getDeviceAction_right(float* retrived_info, int n1);
void getDeviceAction_left(float* retrived_info2, int n2);


#include "INCLUDE/test.h"
