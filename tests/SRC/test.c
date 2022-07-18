/*****************************************************************************

Copyright (c) 2004 SensAble Technologies, Inc. All rights reserved.

OpenHaptics(TM) toolkit. The material embodied in this software and use of
this software is subject to the terms and conditions of the clickthrough
Development License Agreement.

For questions, comments or bug reports, go to forums at: 
    http://dsc.sensable.com

Module Name:

  Calibration.c

Description: 

  This example demonstrates how to handle haptic device calibration using the
  functions available in the HD API.

*******************************************************************************/
#ifdef  _WIN64
#pragma warning (disable:4996)
#endif

#include <stdio.h>
#include <assert.h>

#if defined(WIN32)
# include <windows.h>
# include <test.h>
#else
# include "test.h"
# include <unistd.h>
# define Sleep(x) usleep((x) * 1000)
#endif

#include <HD/hd.h>
#include <HDU/hduError.h>
#include <HDU/hduVector.h>

HDCallbackCode HDCALLBACK DeviceAllInfoCallback_right(void *pUserData);
HDCallbackCode HDCALLBACK DeviceAllInfoCallback_left(void *pUserData);

HDenum GetCalibrationStatus_right();
HDenum GetCalibrationStatus_left();
void getDeviceAction_right(float* retrived_info, int n1);
void getDeviceAction_left(float* retrived_info2, int n2);

struct S_Haptic_info
{
    /* data */
    hduVector3Dd position;
    hduVector3Dd angle;
    HDint button;
};

/*******************************************************************************
 Main function.
*******************************************************************************/
HHD hHD_right;
int calibrationStyle_right;

HHD hHD_left;
int calibrationStyle_left;

int initTouch_right()
{
    HDErrorInfo error;
    int supportedCalibrationStyles;

    hHD_right = hdInitDevice("Right");
    if (HD_DEVICE_ERROR(error = hdGetError())) 
    {
        hduPrintError(stderr, &error, "Failed to initialize haptic device");
        fprintf(stderr, "\nPress any key to quit.\n");
        getch();
        return -1;
    }

    printf("Haptic Calibration\n");
    printf("Found haptic device: %s.\n\n", hdGetString(HD_DEVICE_MODEL_TYPE));

    /* Choose a calibration style.  Some devices may support multiple types of 
       calibration.  In that case, prefer auto calibration over inkwell 
       calibration, and prefer inkwell calibration over reset encoders. */
    hdGetIntegerv(HD_CALIBRATION_STYLE, &supportedCalibrationStyles);
    if (supportedCalibrationStyles & HD_CALIBRATION_ENCODER_RESET)
    {
        calibrationStyle_right = HD_CALIBRATION_ENCODER_RESET;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_INKWELL)
    {
        calibrationStyle_right = HD_CALIBRATION_INKWELL;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_AUTO)
    {
        calibrationStyle_right = HD_CALIBRATION_AUTO;
    }

    /* Some haptic devices only support manual encoder calibration via a
       hardware reset. This requires that the endpoint be placed at a known
       physical location when the reset is commanded. For the PHANTOM haptic
       devices, this means positioning the device so that all links are
       orthogonal. Also, this reset is typically performed before the servoloop
       is running, and only technically needs to be performed once after each
       time the device is plugged in. */
    if (calibrationStyle_right == HD_CALIBRATION_ENCODER_RESET)
    {
        printf("Please prepare for manual calibration by\n");
        printf("placing the device at its reset position.\n\n");
        printf("Press any key to continue...\n");

        getch();

        hdUpdateCalibration(calibrationStyle_right);
        if (hdCheckCalibration() == HD_CALIBRATION_OK)
        {
            printf("Calibration complete.\n\n");
        }
        if (HD_DEVICE_ERROR(error = hdGetError()))
        {
            hduPrintError(stderr, &error, "Reset encoders reset failed.");
            return -1;           
        }
    }

    // hdStartScheduler();
    // if (HD_DEVICE_ERROR(error = hdGetError()))
    // {
    //     hduPrintError(stderr, &error, "Failed to start the scheduler");
    //     return -1;           
    // }

    /* Some haptic devices are calibrated when the gimbal is placed into
       the device inkwell and updateCalibration is called.  This form of
       calibration is always performed after the servoloop has started 
       running. */
    if (calibrationStyle_right  == HD_CALIBRATION_INKWELL)
    {
        if (GetCalibrationStatus_right() == HD_CALIBRATION_NEEDS_MANUAL_INPUT)
        {
            printf("Please place the device into the inkwell ");
            printf("for calibration.\n\n");
        }
    }

    return 0;
}

int initTouch_left()
{
    HDErrorInfo error;
    int supportedCalibrationStyles;

    hHD_left = hdInitDevice("Left");
    if (HD_DEVICE_ERROR(error = hdGetError())) 
    {
        hduPrintError(stderr, &error, "Failed to initialize haptic device");
        fprintf(stderr, "\nPress any key to quit.\n");
        getch();
        return -1;
    }

    printf("Haptic Calibration\n");
    printf("Found haptic device: %s.\n\n", hdGetString(HD_DEVICE_MODEL_TYPE));

    /* Choose a calibration style.  Some devices may support multiple types of 
       calibration.  In that case, prefer auto calibration over inkwell 
       calibration, and prefer inkwell calibration over reset encoders. */
    hdGetIntegerv(HD_CALIBRATION_STYLE, &supportedCalibrationStyles);
    if (supportedCalibrationStyles & HD_CALIBRATION_ENCODER_RESET)
    {
        calibrationStyle_left = HD_CALIBRATION_ENCODER_RESET;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_INKWELL)
    {
        calibrationStyle_left = HD_CALIBRATION_INKWELL;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_AUTO)
    {
        calibrationStyle_left = HD_CALIBRATION_AUTO;
    }

    /* Some haptic devices only support manual encoder calibration via a
       hardware reset. This requires that the endpoint be placed at a known
       physical location when the reset is commanded. For the PHANTOM haptic
       devices, this means positioning the device so that all links are
       orthogonal. Also, this reset is typically performed before the servoloop
       is running, and only technically needs to be performed once after each
       time the device is plugged in. */
    if (calibrationStyle_left == HD_CALIBRATION_ENCODER_RESET)
    {
        printf("Please prepare for manual calibration by\n");
        printf("placing the device at its reset position.\n\n");
        printf("Press any key to continue...\n");

        getch();

        hdUpdateCalibration(calibrationStyle_left);
        if (hdCheckCalibration() == HD_CALIBRATION_OK)
        {
            printf("Calibration complete.\n\n");
        }
        if (HD_DEVICE_ERROR(error = hdGetError()))
        {
            hduPrintError(stderr, &error, "Reset encoders reset failed.");
            return -1;           
        }
    }

    // hdStartScheduler();
    // if (HD_DEVICE_ERROR(error = hdGetError()))
    // {
    //     hduPrintError(stderr, &error, "Failed to start the scheduler");
    //     return -1;           
    // }

    /* Some haptic devices are calibrated when the gimbal is placed into
       the device inkwell and updateCalibration is called.  This form of
       calibration is always performed after the servoloop has started 
       running. */
    if (calibrationStyle_left  == HD_CALIBRATION_INKWELL)
    {
        if (GetCalibrationStatus_left() == HD_CALIBRATION_NEEDS_MANUAL_INPUT)
        {
            printf("Please place the device into the inkwell ");
            printf("for calibration.\n\n");
        }
    }

    return 0;
}

void startScheduler()
{
     HDErrorInfo error;
    hdStartScheduler();
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        hduPrintError(stderr, &error, "Failed to start the scheduler");
        return -1;           
    }
}

void stopScheduler()
{
    hdStopScheduler();
}

void closeTouch_left()
{
    // hdStopScheduler();
    hdDisableDevice(hHD_left);
    return;
}

void closeTouch_right()
{
    // hdStopScheduler();
    hdDisableDevice(hHD_right);
    return;
}

/******************************************************************************
 Begin Scheduler callbacks
 */

HDCallbackCode HDCALLBACK CalibrationStatusCallback_left(void *pUserData)
{
    HDenum *pStatus = (HDenum *) pUserData;

    hdBeginFrame(hHD_left);
    *pStatus = hdCheckCalibration();
    hdEndFrame(hHD_left);

    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK CalibrationStatusCallback_right(void *pUserData)
{
    HDenum *pStatus = (HDenum *) pUserData;

    hdBeginFrame(hHD_right);
    *pStatus = hdCheckCalibration();
    hdEndFrame(hHD_right);

    return HD_CALLBACK_DONE;
}


HDCallbackCode HDCALLBACK DeviceAllInfoCallback_right(void *pUserData)
{

    struct S_Haptic_info *pinfo = (struct S_Haptic_info *) pUserData;

    hduVector3Dd pre_posistion;
    hduVector3Dd pre_angle;

    hdBeginFrame(hHD_right);

    hdGetDoublev(HD_LAST_POSITION, &pre_posistion);
    hdGetDoublev(HD_LAST_GIMBAL_ANGLES, &pre_angle);

    hdGetIntegerv(HD_CURRENT_BUTTONS, &(pinfo->button));
    hdGetDoublev(HD_CURRENT_POSITION, &(pinfo->position));
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, &(pinfo->angle));

    hdEndFrame(hHD_right);

    pinfo->position[0] = pinfo->position[0] - pre_posistion[0];
    pinfo->position[1] = pinfo->position[1] - pre_posistion[1];
    pinfo->position[2] = pinfo->position[2] - pre_posistion[2];

    pinfo->angle[0] = pinfo->angle[0] - pre_angle[0];
    pinfo->angle[1] = pinfo->angle[1] - pre_angle[1];
    pinfo->angle[2] = pinfo->angle[2] - pre_angle[2];

    // printf("Device left position: %.3f %.3f %.3f\n", 
    //     pre_posistion[0], pre_posistion[1], pre_posistion[2]);
    // printf("Device left angle: %.3f %.3f %.3f\n", 
    //     pinfo->angle[0], pinfo->angle[1], pinfo->angle[2]);
    // printf("Device left button: %d\n", pinfo->button);

    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK DeviceAllInfoCallback_left(void *pUserData)
{

    struct S_Haptic_info *pinfo = (struct S_Haptic_info *) pUserData;

    hduVector3Dd pre_posistion;
    hduVector3Dd pre_angle;

    hdBeginFrame(hHD_left);

    hdGetDoublev(HD_LAST_POSITION, &pre_posistion);
    hdGetDoublev(HD_LAST_GIMBAL_ANGLES, &pre_angle);

    hdGetIntegerv(HD_CURRENT_BUTTONS, &(pinfo->button));
    hdGetDoublev(HD_CURRENT_POSITION, &(pinfo->position));
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, &(pinfo->angle));

    hdEndFrame(hHD_left);

    pinfo->position[0] = pinfo->position[0] - pre_posistion[0];
    pinfo->position[1] = pinfo->position[1] - pre_posistion[1];
    pinfo->position[2] = pinfo->position[2] - pre_posistion[2];

    pinfo->angle[0] = pinfo->angle[0] - pre_angle[0];
    pinfo->angle[1] = pinfo->angle[1] - pre_angle[1];
    pinfo->angle[2] = pinfo->angle[2] - pre_angle[2];

    // printf("Device left position: %.3f %.3f %.3f\n", 
    //     pinfo->position[0], pinfo->position[1], pinfo->position[2]);
    // printf("Device left angle: %.3f %.3f %.3f\n", 
    //     pinfo->angle[0], pinfo->angle[1], pinfo->angle[2]);
    // printf("Device left button: %d\n", pinfo->button);

    return HD_CALLBACK_DONE;
}

HDenum GetCalibrationStatus_left()
{
    HDenum status;
    hdScheduleSynchronous(CalibrationStatusCallback_left, &status,
                          HD_DEFAULT_SCHEDULER_PRIORITY);
    return status;
}

HDenum GetCalibrationStatus_right()
{
    HDenum status;
    hdScheduleSynchronous(CalibrationStatusCallback_right, &status,
                          HD_DEFAULT_SCHEDULER_PRIORITY);
    return status;
}

int count_frame_right = 0;
int count_frame_left = 0;

void getDeviceAction_right(float* retrived_info, int n1)
{

    struct S_Haptic_info myinfo;

    hdScheduleSynchronous(DeviceAllInfoCallback_right, &myinfo,
    HD_DEFAULT_SCHEDULER_PRIORITY);
    if (count_frame_right<=1)
    {
        retrived_info[0] = 0;
        retrived_info[1] = 0;
        retrived_info[2] = 0;
        retrived_info[3] = 0;
        retrived_info[4] = myinfo.button;   
    }
    else{
        retrived_info[0] = myinfo.position[0];
        retrived_info[1] = myinfo.position[1];
        retrived_info[2] = myinfo.position[2];
        retrived_info[3] = myinfo.angle[2];
        retrived_info[4] = myinfo.button;        
    }
    count_frame_right= count_frame_right + 1;    

    // printf("Device right position: %.3f %.3f %.3f\n", 
    //     myinfo.position[0], myinfo.position[1], myinfo.position[2]);
    // printf("Device right angle: %.3f %.3f %.3f\n", 
    //     myinfo.angle[0], myinfo.angle[1], myinfo.angle[2]);
    // printf("Device right button: %d\n", myinfo.button);
}

void getDeviceAction_left(float* retrived_info2, int n2)
{
    struct S_Haptic_info myinfo;

    hdScheduleSynchronous(DeviceAllInfoCallback_left, &myinfo,
    HD_DEFAULT_SCHEDULER_PRIORITY);
    if (count_frame_left<=1)
    {
        retrived_info2[0] = 0;
        retrived_info2[1] = 0;
        retrived_info2[2] = 0;
        retrived_info2[3] = 0;
        retrived_info2[4] = myinfo.button;   
    }
    else{
        retrived_info2[0] = myinfo.position[0];
        retrived_info2[1] = myinfo.position[1];
        retrived_info2[2] = myinfo.position[2];
        retrived_info2[3] = myinfo.angle[2];
        retrived_info2[4] = myinfo.button;        
    }
    count_frame_left= count_frame_left + 1;    

    // printf("Device left position: %.3f %.3f %.3f\n", 
    //     myinfo.position[0], myinfo.position[1], myinfo.position[2]);
    // printf("Device left angle: %.3f %.3f %.3f\n", 
    //     myinfo.angle[0], myinfo.angle[1], myinfo.angle[2]);
    // printf("Device left button: %d\n", myinfo.button);
}
/*
 End Scheduler callbacks
 *****************************************************************************/

/*****************************************************************************/

