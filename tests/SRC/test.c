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

HDCallbackCode HDCALLBACK UpdateCalibrationCallback(void *pUserData);
HDCallbackCode HDCALLBACK CalibrationStatusCallback(void *pUserData);
HDCallbackCode HDCALLBACK DevicePositionCallback(void *pUserData);
HDCallbackCode HDCALLBACK DeviceAngleCallback(void *pUserData);
HDCallbackCode HDCALLBACK DeviceButtonCallback(void *pUserData);
HDCallbackCode HDCALLBACK DeviceAllInfoCallback(void *pUserData);

HDenum GetCalibrationStatus();
HDboolean CheckCalibration(HDenum calibrationStyle);
void PrintDevicePosition();
void getDeviceAction(float* retrived_info, int n1);

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
HHD hHD;
int calibrationStyle;

int TESTtouch()
{
    //HHD hHD;
    HDErrorInfo error;
    int supportedCalibrationStyles;

    hHD = hdInitDevice(HD_DEFAULT_DEVICE);
    if (HD_DEVICE_ERROR(error = hdGetError())) 
    {
        hduPrintError(stderr, &error, "Failed to initialize haptic device");
        fprintf(stderr, "\nPress any key to quit.\n");
        getch();
        return -1;
    }

    printf("Calibration\n");
    printf("Found %s.\n\n", hdGetString(HD_DEVICE_MODEL_TYPE));

    /* Choose a calibration style.  Some devices may support multiple types of 
       calibration.  In that case, prefer auto calibration over inkwell 
       calibration, and prefer inkwell calibration over reset encoders. */
    hdGetIntegerv(HD_CALIBRATION_STYLE, &supportedCalibrationStyles);
    if (supportedCalibrationStyles & HD_CALIBRATION_ENCODER_RESET)
    {
        calibrationStyle = HD_CALIBRATION_ENCODER_RESET;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_INKWELL)
    {
        calibrationStyle = HD_CALIBRATION_INKWELL;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_AUTO)
    {
        calibrationStyle = HD_CALIBRATION_AUTO;
    }

    /* Some haptic devices only support manual encoder calibration via a
       hardware reset. This requires that the endpoint be placed at a known
       physical location when the reset is commanded. For the PHANTOM haptic
       devices, this means positioning the device so that all links are
       orthogonal. Also, this reset is typically performed before the servoloop
       is running, and only technically needs to be performed once after each
       time the device is plugged in. */
    if (calibrationStyle == HD_CALIBRATION_ENCODER_RESET)
    {
        printf("Please prepare for manual calibration by\n");
        printf("placing the device at its reset position.\n\n");
        printf("Press any key to continue...\n");

        getch();

        hdUpdateCalibration(calibrationStyle);
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

    hdStartScheduler();
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        hduPrintError(stderr, &error, "Failed to start the scheduler");
        return -1;           
    }

    /* Some haptic devices are calibrated when the gimbal is placed into
       the device inkwell and updateCalibration is called.  This form of
       calibration is always performed after the servoloop has started 
       running. */
    if (calibrationStyle  == HD_CALIBRATION_INKWELL)
    {
        if (GetCalibrationStatus() == HD_CALIBRATION_NEEDS_MANUAL_INPUT)
        {
            printf("Please place the device into the inkwell ");
            printf("for calibration.\n\n");
        }
    }

    printf("Press any key to quit.\n\n");

    /* Loop until key press. */
    while (!_kbhit())
    {
        /* Regular calibration should be checked periodically while the
           servoloop is running. In some cases, like the PHANTOM Desktop,
           calibration can be continually refined as the device is moved
           throughout its workspace.  For other devices that require inkwell
           reset, such as the PHANToM Omni, calibration is successfully
           performed whenever the device is put into the inkwell. */
        if (CheckCalibration(calibrationStyle))
        {
            PrintDevicePosition();
        }

        Sleep(1000);
    }
    
    hdStopScheduler();
    hdDisableDevice(hHD);

    return 0;
}


int initTouch()
{
    HDErrorInfo error;
    int supportedCalibrationStyles;

    hHD = hdInitDevice(HD_DEFAULT_DEVICE);
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
        calibrationStyle = HD_CALIBRATION_ENCODER_RESET;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_INKWELL)
    {
        calibrationStyle = HD_CALIBRATION_INKWELL;
    }
    if (supportedCalibrationStyles & HD_CALIBRATION_AUTO)
    {
        calibrationStyle = HD_CALIBRATION_AUTO;
    }

    /* Some haptic devices only support manual encoder calibration via a
       hardware reset. This requires that the endpoint be placed at a known
       physical location when the reset is commanded. For the PHANTOM haptic
       devices, this means positioning the device so that all links are
       orthogonal. Also, this reset is typically performed before the servoloop
       is running, and only technically needs to be performed once after each
       time the device is plugged in. */
    if (calibrationStyle == HD_CALIBRATION_ENCODER_RESET)
    {
        printf("Please prepare for manual calibration by\n");
        printf("placing the device at its reset position.\n\n");
        printf("Press any key to continue...\n");

        getch();

        hdUpdateCalibration(calibrationStyle);
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

    hdStartScheduler();
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        hduPrintError(stderr, &error, "Failed to start the scheduler");
        return -1;           
    }

    /* Some haptic devices are calibrated when the gimbal is placed into
       the device inkwell and updateCalibration is called.  This form of
       calibration is always performed after the servoloop has started 
       running. */
    if (calibrationStyle  == HD_CALIBRATION_INKWELL)
    {
        if (GetCalibrationStatus() == HD_CALIBRATION_NEEDS_MANUAL_INPUT)
        {
            printf("Please place the device into the inkwell ");
            printf("for calibration.\n\n");
        }
    }

    return 0;
}

void closeTouch()
{
    hdStopScheduler();
    hdDisableDevice(hHD);
    return;
}

/******************************************************************************
 Begin Scheduler callbacks
 */

HDCallbackCode HDCALLBACK UpdateCalibrationCallback(void *pUserData)
{
    HDenum *calibrationStyle = (int *) pUserData;

    if (hdCheckCalibration() == HD_CALIBRATION_NEEDS_UPDATE)
    {
        hdUpdateCalibration(*calibrationStyle);
    }

    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK CalibrationStatusCallback(void *pUserData)
{
    HDenum *pStatus = (HDenum *) pUserData;

    hdBeginFrame(hdGetCurrentDevice());
    *pStatus = hdCheckCalibration();
    hdEndFrame(hdGetCurrentDevice());

    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK DevicePositionCallback(void *pUserData)
{
    HDdouble *pPosition = (HDdouble *) pUserData;

    hdBeginFrame(hdGetCurrentDevice());
    hdGetDoublev(HD_CURRENT_POSITION, pPosition);
    hdEndFrame(hdGetCurrentDevice());

    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK DeviceAngleCallback(void *pUserData)
{
    HDdouble *pAngle = (HDdouble *) pUserData;

    hdBeginFrame(hdGetCurrentDevice());
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, pAngle);
    hdEndFrame(hdGetCurrentDevice());

    return HD_CALLBACK_DONE;
}
HDCallbackCode HDCALLBACK DeviceButtonCallback(void *pUserData)
{
    HDint *pButton = (HDint *) pUserData;

    hdBeginFrame(hdGetCurrentDevice());
    hdGetIntegerv(HD_CURRENT_BUTTONS, pButton);
    hdEndFrame(hdGetCurrentDevice());

    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK DeviceAllInfoCallback(void *pUserData)
{

    struct S_Haptic_info *pinfo = (struct S_Haptic_info *) pUserData;

    hduVector3Dd pre_posistion;

    hdBeginFrame(hdGetCurrentDevice());

    hdGetDoublev(HD_LAST_POSITION, &pre_posistion);

    hdGetIntegerv(HD_CURRENT_BUTTONS, &(pinfo->button));
    hdGetDoublev(HD_CURRENT_POSITION, &(pinfo->position));
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, &(pinfo->angle));

    hdEndFrame(hdGetCurrentDevice());

    pinfo->position[0] = pinfo->position[0] - pre_posistion[0];
    pinfo->position[1] = pinfo->position[1] - pre_posistion[1];
    pinfo->position[2] = pinfo->position[2] - pre_posistion[2];

    return HD_CALLBACK_DONE;
}

HDenum GetCalibrationStatus()
{
    HDenum status;
    hdScheduleSynchronous(CalibrationStatusCallback, &status,
                          HD_DEFAULT_SCHEDULER_PRIORITY);
    return status;
}

HDboolean CheckCalibration(HDenum calibrationStyle)
{
    HDenum status = GetCalibrationStatus();
    
    if (status == HD_CALIBRATION_OK)
    {
        return HD_TRUE;
    }
    else if (status == HD_CALIBRATION_NEEDS_MANUAL_INPUT)
    {
        printf("Calibration requires manual input...\n");
        return HD_FALSE;
    }
    else if (status == HD_CALIBRATION_NEEDS_UPDATE)
    {
        hdScheduleSynchronous(UpdateCalibrationCallback, &calibrationStyle,
            HD_DEFAULT_SCHEDULER_PRIORITY);

        if (HD_DEVICE_ERROR(hdGetError()))
        {
            printf("\nFailed to update calibration.\n\n");
            return HD_FALSE;
        }
        else
        {
            printf("\nCalibration updated successfully.\n\n");
            return HD_TRUE;
        }
    }
    else
    {
        assert(!"Unknown calibration status");
        return HD_FALSE;
    }
}

void PrintDevicePosition()
{
    struct S_Haptic_info myinfo;

    hdScheduleSynchronous(DeviceAllInfoCallback, &myinfo,
        HD_DEFAULT_SCHEDULER_PRIORITY);
        
    printf("Device position: %.3f %.3f %.3f\n", 
        myinfo.position[0], myinfo.position[1], myinfo.position[2]);
    printf("Device angle: %.3f %.3f %.3f\n", 
        myinfo.angle[0], myinfo.angle[1], myinfo.angle[2]);
    printf("Device button: %d\n", myinfo.button);
}

int count_frame = 0;
void getDeviceAction(float* retrived_info, int n1)
{

    struct S_Haptic_info myinfo;

    hdScheduleSynchronous(DeviceAllInfoCallback, &myinfo,
        HD_DEFAULT_SCHEDULER_PRIORITY);
        
    // printf("Device position: %.3f %.3f %.3f\n", 
    //     myinfo.position[0], myinfo.position[1], myinfo.position[2]);
    // printf("Device angle: %.3f %.3f %.3f\n", 
    //     myinfo.angle[0], myinfo.angle[1], myinfo.angle[2]);
    // printf("Device button: %d\n", myinfo.button);
    if (count_frame<=1)
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
        retrived_info[3] = 0;
        retrived_info[4] = myinfo.button;        
    }
    count_frame= count_frame + 1;


}

/*
 End Scheduler callbacks
 *****************************************************************************/

/*****************************************************************************/
