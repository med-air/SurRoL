/*****************************************************************************

Copyright (c) 2005 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com
                                          
Module Name:

    conio.h

Description: 

    Console functionality required by example code.

******************************************************************************/

#ifndef __CONIO_H_
#define __CONIO_H_

#ifdef _cplusplus
extern "C" {
#endif // _cplusplus

int _kbhit();
int getch();

#ifdef _cplusplus
}
#endif // _cplusplus

#endif // __CONIO_H

/******************************************************************************/
