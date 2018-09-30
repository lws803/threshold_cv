//
//  main.cpp
//  k_nearest_detector_v2
//
//  Created by Ler Wilson on 11/9/18.
//  Copyright © 2018 Ler Wilson. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#define INT_INF 2147483640

using namespace cv;
using namespace std;

/**
 * Three parameters to tune
 * COLOR_SELECT lets you choose what color you wish to target
 * DISTANCE_DIFFERENCE determines the max distance you wish to stretch the color space (makes colors more distinct)
 * DISTANCE_LIMIT_FILTER additional filter to make sure that we only perform adaptive if it is below a certain distance
 * OUTPUT_MODE determines which output you wish to view
 * MIN_HARDSTOP determines the cut off before no masking is shown at all
*/


float MIN_GLOBAL, MAX_GLOBAL;
string COLOR_SELECT = "CUSTOM";
float CLIMB = 0.1;

class ColorMap {
    vector<float> colorArray = {0,0,0};
    int DISTANCE_DIFFERENCE = 1000;
    float DISTANCE_LIMIT_FILTER = 370; // More means larger allowance (max is 370)
    float MIN_HARDSTOP = 50; // More means more leeway. Use this to stop detection if color you want is really not in view
public:
    ColorMap (string COLOR_SELECT) {
        if (COLOR_SELECT == "PURE_RED") {
            colorArray = {54.29/100 * 255, 80.81 + 127, 69.89 + 127};
        }
        else if (COLOR_SELECT == "PURE_GREEN") {
            colorArray = {46.228/100 * 255, -51.699 + 127,  49.897 + 127};
        }
        else if (COLOR_SELECT == "PURE_BLUE") {
            colorArray = {29.57/100 * 255, 68.30 + 127,  -112.03 + 127};
        }
        else if (COLOR_SELECT == "PURE_YELLOW") {
            colorArray = {97.139/100 * 255, -21.558 + 127,  94.477 + 127};
        }
        else if (COLOR_SELECT == "DARK_BLUE") {
            colorArray = {56.14306376, 163.50908374, 72.79013131};
            MIN_HARDSTOP = 17;
        }
        else if (COLOR_SELECT == "SELECTIVE_PURE_RED") {
            /**
             * Steps:
             * 1. First we find the minimum distance when red isnt present
             * 2. We set that as DISTANCE_LIMIT_FILTER to block it out
             * 3. We slowly tune DISTANCE_DIFFERENCE by lowering it so that the full range of our red is present when in
             * view
             */
            colorArray = {54.29/100 * 255, 80.81 + 127, 69.89 + 127};
            DISTANCE_DIFFERENCE = 200;
            DISTANCE_LIMIT_FILTER = 67;
        }
        else if (COLOR_SELECT == "CUSTOM") {
            colorArray = {157.15862154, 118.09701874, 177.16864464};
        }
        else {
            colorArray = {97.139/100 * 255, -21.558 + 127,  94.477 + 127}; // PURE_YELLOW
        }
    }
    
    vector<float> getColor () {
        return colorArray;
    }
    int getDistanceDifference () {
        return DISTANCE_DIFFERENCE;
    }
    float getDistanceLimitFilter () {
        return DISTANCE_LIMIT_FILTER;
    }
    float getMinHardStop () {
        return MIN_HARDSTOP;
    }
};


class pipeline {
    Mat inputImg, processed, mask, preprocessed;
    ColorMap myColorChoice = ColorMap(COLOR_SELECT);

    float MULTIPLIER = 10;
    float REAL_MIN = INT_INF;
    
    enum FUNCTION_TYPE {
        QUADRATIC, MULTIPLICATIVE
    };
    enum OUTPUT_MODE {
        MASKED, PROCESSED, PREPROCESSED
    };
    
    FUNCTION_TYPE FUNCTION = MULTIPLICATIVE;
    OUTPUT_MODE OUTPUT = PROCESSED;
    
    
    float cartesian_dist (vector<float> colorArray, vector<uchar> lab_channels) {
//        float difference_1 = pow((colorArray[0] - lab_channels[0]), 2);
        float difference_2 = pow((colorArray[1] - lab_channels[1]), 2);
        float difference_3 = pow((colorArray[2] - lab_channels[2]), 2);
        return sqrt(difference_2 + difference_3);
    }
    
    Mat preprocessor (Mat input) {
        Mat lab_image = input.clone();
        
        // Gaussian blurring
//        GaussianBlur(input, lab_image, Size( 5, 5 ), 0, 0);
        cvtColor(lab_image, lab_image, CV_BGR2Lab);
        return lab_image;
    }
    
    float multiply_dist (float dist) {
        switch (FUNCTION) {
            case QUADRATIC:
                dist = pow(dist, MULTIPLIER);
                break;
            case MULTIPLICATIVE:
                dist *= MULTIPLIER;
                break;
            default:
                break;
        }
        return dist;
    }
    
    
    float cut_off_dist (float dist) {
        if (dist > 255) dist = 255;
        if (dist < 0) dist = 0;
        
        return dist;
    }
    
    
    Mat k_nearest (Mat lab_image) {
        Mat LAB[3];
        split(lab_image, LAB);

        
        Mat img(lab_image.rows, lab_image.cols, CV_8UC1, Scalar(0));
        
        float min = INT_INF;
        float max = 0;
        
        // Main processing
        for (int i = 0; i < lab_image.rows; i++) {
            for (int d = 0; d < lab_image.cols; d++) {
                vector<uchar> lab_channels = {LAB[0].at<uchar>(i, d),
                    LAB[1].at<uchar>(i, d),
                    LAB[2].at<uchar>(i, d)};
                
                float dist = cartesian_dist(myColorChoice.getColor(), lab_channels);
                
                
                if (dist >= myColorChoice.getDistanceLimitFilter()) {
                    // If it is beyond the distance we want then just ignore.
                    dist = 255; // Make it max
                }else{
                    if (dist < REAL_MIN) REAL_MIN = dist;

                    dist = multiply_dist(dist);
                    if (dist < min) min = dist;
                    if (dist > max) max = dist;
                    
                    dist -= MIN_GLOBAL;
                    
                    dist = cut_off_dist(dist);
                }                
                img.at<uchar>(i, d) = 255 - round(dist);
            }
        }
        
        // TODO: Double check this
        if (REAL_MIN > myColorChoice.getMinHardStop()) {
            // Ignore everything
            img = Mat(lab_image.rows, lab_image.cols, CV_8UC1, Scalar(0));
        } else {
            MAX_GLOBAL = max;
            MIN_GLOBAL = min;
            
            autoAdjust(max, min);
        }
        
        return img;
    }
    
    void autoAdjust (float max, float min) {
        if (max - min < myColorChoice.getDistanceDifference()) MULTIPLIER += CLIMB;
        if (max - min > myColorChoice.getDistanceDifference()) MULTIPLIER -= CLIMB;
        if (MULTIPLIER < 1) MULTIPLIER = 1;
        if (MULTIPLIER > 10000) MULTIPLIER = 10000;
    }
    
    
    Mat threshold (Mat input) {
        Mat output;
        inRange(input, Scalar(20), Scalar(255), output);
        return output;
    }
    
    
public:
    pipeline (Mat input, float multiplier) {
        // constructor
        time_t start, end;
        time(&start);
        this->MULTIPLIER = multiplier;
        this->inputImg = input;
        this->preprocessed = preprocessor(this->inputImg);
        this->processed = k_nearest(this->preprocessed);
        this->mask = threshold(this->processed);
        
        time(&end);
        double seconds = difftime (end, start);
        
//        cout << (double)seconds << endl;
    }
    
    Mat visualise () {
        switch (OUTPUT) {
            case MASKED:
                return this->mask;
                break;
            case PROCESSED:
                return this->processed;
                break;
            case PREPROCESSED:
                return this->preprocessed;
                break;
            default:
                return this->inputImg;
        }
    }
    
    float getProposedMultipler () {
        return this->MULTIPLIER;
    }
    
    float getRealMin () {
        return this->REAL_MIN;
    }
};


int main(int argc, char ** argv) {
    VideoCapture cap(0); // webcam
    Mat source;
    float multiplier = 10;
    
    vector<std::string> args(argv, argv + argc);
    if (argc > 1) {
        cout << args[1] << endl;
        COLOR_SELECT = args[1];
    }
    
    while (true) {
        Mat output;
        cap.read(source);
        resize(source, source, Size(), 0.25, 0.25);
        pipeline myPipeline = pipeline(source, multiplier);

        namedWindow("My Window", WINDOW_AUTOSIZE);
        multiplier = myPipeline.getProposedMultipler(); // To auto adjust
//        cout << myPipeline.getRealMin() << endl;

        resize(myPipeline.visualise(), output, Size(), 2, 2); // upscale it up
        
        imshow("My Window", output);
        waitKey(1);
    }
    return 0;
}
