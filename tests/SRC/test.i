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


int TESTtouch();

int initTouch();
void closeTouch();
void getDeviceAction(float* retrived_info, int n1);


#include "INCLUDE/test.h"
