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

using namespace cv;
using namespace std;

class pipeline {
    Mat inputImg, processed;
    
    float cartesian_dist (vector<float> colorArray, vector<uchar> lab_channels) {
//        float difference_1 = pow((colorArray[0] - lab_channels[0]), 2);
        float difference_2 = pow((colorArray[1] - lab_channels[1]), 2);
        float difference_3 = pow((colorArray[2] - lab_channels[2]), 2);
        return sqrt(difference_2 + difference_3);
    }

    
    Mat preprocessor () {
        Mat lab_image, LAB[3];
        vector<float> colorArray = {54.29/100 * 255, 80.81 + 127, 69.89 + 127};
        vector<float> colorArray_green = {87.82/100 * 255, -79.29 + 127,  80.99 + 127};
        
        cvtColor(this->inputImg, lab_image, CV_BGR2Lab);
        split(lab_image, LAB);
        Mat img(lab_image.rows, lab_image.cols, CV_8UC1, Scalar(0));
        
        
        // Main processing
        for (int i = 0; i < lab_image.rows; i++) {
            for (int d = 0; d < lab_image.cols; d++) {
                vector<uchar> lab_channels = {LAB[0].at<uchar>(i, d),
                    LAB[1].at<uchar>(i, d),
                    LAB[2].at<uchar>(i, d)};
                float dist = cartesian_dist(colorArray, lab_channels);
                img.at<uchar>(i, d) = 255 - round(dist);
            }
        }
        return img;
    }
    
    Mat threshold (Mat input) {
        // TODO: Add thresholding here
        return input;
    }
    
    
public:
    pipeline (Mat input) {
        // constructor
        time_t start, end;
        time(&start);
        
        this->inputImg = input;
        this->processed = preprocessor();
        
        time(&end);
        double seconds = difftime (end, start);
        
        cout << (double)seconds << endl;
    }
    
    Mat visualise () {
        return this->processed;
    }
};


int main(int argc, const char * argv[]) {
    VideoCapture cap(0); // webcam
    Mat source;
    while (true) {
        cap.read(source);
        resize(source, source, Size(), 0.3, 0.3);
        pipeline myPipeline = pipeline(source);
        
        namedWindow("My Window", WINDOW_AUTOSIZE);
        //imshow("Viewfinder", viewfinder);
        resize(myPipeline.visualise(), source, Size(), 2, 2); // upscale it up
        imshow("My Window", source);
        
        waitKey(1);

    }
    return 0;
}
