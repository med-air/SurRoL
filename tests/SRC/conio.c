 /*****************************************************************************

Copyright (c) 2005 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com
                                          
Module Name:

    conio.c

Description: 

    Console functionality required by example code.

******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <termios.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

static struct termios term_attribs, term_attribs_old;

static void restore_term(void)
{
    if(tcsetattr(STDIN_FILENO, TCSAFLUSH, &term_attribs_old) < 0)
    {
        perror("tcsetattr: ");
        exit(-1);
    }
}

int _kbhit()
{
    static int initialized;
	
    fd_set rfds;
    struct timeval tv;
    int retcode;
	
    if(!initialized)
    {
        if(tcgetattr(STDIN_FILENO, &term_attribs) < 0)
        {
            perror("tcgetattr: ");
            exit(-1);
        }

        term_attribs_old = term_attribs;

        if(atexit(restore_term))
        {
            perror("atexit: ");
            exit(-1);
        }

        term_attribs.c_lflag &= ~(ECHO | ICANON | ISIG | IEXTEN);
        term_attribs.c_iflag &= ~(IXON | BRKINT | INPCK | ICRNL | ISTRIP);
        term_attribs.c_cflag &= ~(CSIZE | PARENB);
        term_attribs.c_cflag |= CS8;
        term_attribs.c_cc[VTIME] = 0;
        term_attribs.c_cc[VMIN] = 0;

        if(tcsetattr(STDIN_FILENO, TCSANOW, &term_attribs) < 0)
        {
            perror("tcsetattr: ");
            exit(-1);
        }

        initialized = 1;
    }	

    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    memset(&tv, 0, sizeof(tv));
    
    retcode = select(1, &rfds, NULL, NULL, &tv);
    if(retcode == -1 && errno == EINTR)
    {
        return 0;
    }
    else if(retcode < 0)
    {
        perror("select: ");
        exit(-1);
    }
    else if(FD_ISSET(STDIN_FILENO, &rfds))
    {
        return retcode;
    }
	
    return 0;
}

int getch()
{
    fd_set rfds;
    int retcode;

    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    
    retcode = select(1, &rfds, NULL, NULL, NULL);
    if(retcode == -1 && errno == EINTR)
    {
        return 0;
    }
    else if(retcode < 0)
    {
        perror("select: ");
        exit(-1);
    }

    return getchar();
}

