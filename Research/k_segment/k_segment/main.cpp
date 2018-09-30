//
//  main.cpp
//  k_segment
//
//  Created by Ler Wilson on 28/9/18.
//  Copyright © 2018 Ler Wilson. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <map>
#include <unordered_map>

#define INT_INF 2147483640

using namespace cv;
using namespace std;

class Colors {
    unordered_map<string, vector<float> > colors;
    unordered_map<string, vector<float> > representativeColors;
    unordered_map<string, float> thresholds;
    vector<string> names = {"red", "yellow", "blue"};
    
public:
    Colors () {
        colors["red"] = {138.4, 207.81, 196.89};
        colors["yellow"] = {175.14135491, 116.86378348, 180.24186594};
        colors["blue"] = {75.4, 195.3, 14,97};
        
        representativeColors["red"] = {0, 0, 255};
        representativeColors["yellow"] = {0, 255, 255};
        representativeColors["blue"] = {255, 0, 0};
        
        thresholds["red"] = 65;
        thresholds["yellow"] = 20;
        thresholds["blue"] = 90;
    }
    
    int getNumColors () {
        return names.size();
    }
    vector<float> getColorFromIndex (int index) {
        return colors[names[index]];
    }
    float getThreshold (int index) {
        return thresholds [names[index]];
    }
    vector<float> getRepresentativeColorFromName (string name) {
        return representativeColors[name];
    }
    string getNameFromIndex (int index) {
        return names[index];
    }
};

Colors colorBank;

class Pipeline {
    Mat inputImg, processed, preprocessed;
    
    float REAL_MIN = INT_INF;
    
    
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
    
    
    float cut_off_dist (float dist) {
        if (dist > 255) dist = 255;
        if (dist < 0) dist = 0;
        
        return dist;
    }
    
    
    Mat k_nearest (Mat lab_image) {
        Mat LAB[3];
        split(lab_image, LAB);
        
        
        Mat img(lab_image.rows, lab_image.cols, CV_8UC3, Scalar(0,0,0));
        
        // Main processing
        for (int i = 0; i < lab_image.rows; i++) {
            for (int d = 0; d < lab_image.cols; d++) {
                vector<uchar> lab_channels = {LAB[0].at<uchar>(i, d),
                    LAB[1].at<uchar>(i, d),
                    LAB[2].at<uchar>(i, d)};
                
                map<float, string> colorMap;
                
                for (int c = 0; c < colorBank.getNumColors(); c++) {
                    vector<float> color = colorBank.getColorFromIndex(c);
                    float dist = cartesian_dist(color, lab_channels);
                    if (dist < colorBank.getThreshold(c)) {
                        colorMap[dist] = colorBank.getNameFromIndex(c);
                    }
                }
                
                Vec3b currColor;
                
                if (colorMap.empty()) {
                    currColor[0] = 0;
                    currColor[1] = 0;
                    currColor[2] = 0;
                } else {
                    currColor[0] = colorBank.getRepresentativeColorFromName(colorMap.begin()->second)[0];
                    currColor[1] = colorBank.getRepresentativeColorFromName(colorMap.begin()->second)[1];
                    currColor[2] = colorBank.getRepresentativeColorFromName(colorMap.begin()->second)[2];
                }
                
                img.at<Vec3b>(Point(d,i)) = currColor;
            }
        }
        return img;
    }
    
public:
    Pipeline (Mat input) {
        // constructor
        time_t start, end;
        time(&start);
        this->inputImg = input;
        this->preprocessed = preprocessor(this->inputImg);
        this->processed = k_nearest(this->preprocessed);
        
        time(&end);
        double seconds = difftime (end, start);
        
        //        cout << (double)seconds << endl;
    }
    
    Mat visualise () {
        return this->processed;
    }
    
    float getRealMin () {
        return this->REAL_MIN;
    }
};


int main(int argc, char ** argv) {
    VideoCapture cap(0); // webcam
    Mat source;
    
    vector<std::string> args(argv, argv + argc);
    while (true) {
        Mat output;
        cap.read(source);
        resize(source, source, Size(), 0.25, 0.25);
        Pipeline myPipeline = Pipeline(source);
        
        namedWindow("My Window", WINDOW_AUTOSIZE);
        resize(myPipeline.visualise(), output, Size(), 2, 2); // upscale it up
        
        imshow("My Window", output);
        waitKey(1);
    }
    return 0;
}
