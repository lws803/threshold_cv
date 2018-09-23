//
//  main.cpp
//  k_image_enhancer
//
//  Created by Ler Wilson on 24/9/18.
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

// TODO: Experiment and write proper methods for the filtering

// BGR
vector<vector<float> > colors = {
    {29.57/100 * 255, 68.30 + 127, -112.03 + 127},
    {87.82/100 * 255, -79.29 + 127, 80.99 + 127},
    {54.29/100 * 255, 80.81 + 127, 69.89 + 127}
};

class pipeline {
    Mat inputImg, processed, preprocessed;
    
    
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
        Mat BGR[3];
        Mat outputImg;
        split(lab_image, LAB);
        split(this->inputImg, BGR);

        Mat img_b (lab_image.rows, lab_image.cols, CV_8UC1, Scalar(0,0,0));
        Mat img_g (lab_image.rows, lab_image.cols, CV_8UC1, Scalar(0,0,0));
        Mat img_r (lab_image.rows, lab_image.cols, CV_8UC1, Scalar(0,0,0));

        // Main processing
        for (int i = 0; i < lab_image.rows; i++) {
            for (int d = 0; d < lab_image.cols; d++) {
                vector<uchar> lab_channels = {
                    LAB[0].at<uchar>(i, d),
                    LAB[1].at<uchar>(i, d),
                    LAB[2].at<uchar>(i, d)};
                
                float dist_b = cartesian_dist(colors[0], lab_channels);
                float dist_g = cartesian_dist(colors[1], lab_channels);
                float dist_r = cartesian_dist(colors[2], lab_channels);

                
                
                // Weird color cut off
//                if(dist_b < dist_g && dist_b < dist_r)
//                {
//                    img_b.at<uchar>(i, d) = (362 - dist_b)/362 *255;
//                }
//                else if(dist_g < dist_r)
//                {
//                    img_g.at<uchar>(i, d) = (362 - dist_g)/362 *255;
//                }
//                else
//                {
//                    img_r.at<uchar>(i, d) = (362 - dist_r)/362 *255;
//                }

                /**
                 * Normal color filtering
                 * Using distances as a multiplier to select channels RGB
                 */
//                img_b.at<uchar>(i, d) = (362 - dist_b)/362 *255;
//                img_g.at<uchar>(i, d) = (362 - dist_g)/362 *255;
//                img_r.at<uchar>(i, d) = (362 - dist_r)/362 *255;
                
                
                // Selective filtering
                img_b.at<uchar>(i, d) = BGR[0].at<uchar>(i, d);
                img_g.at<uchar>(i, d) = BGR[1].at<uchar>(i, d);
                img_r.at<uchar>(i, d) = BGR[2].at<uchar>(i, d);

                if (dist_b < 50) {
                    img_b.at<uchar>(i, d) *= 1.1;
                    if (img_b.at<uchar>(i, d) >= 255) img_b.at<uchar>(i, d) = 254;
                }

                if (dist_g < 50) {
                    img_g.at<uchar>(i, d)  *= 1.1;
                    if (img_g.at<uchar>(i, d) >= 255) img_g.at<uchar>(i, d) = 254;
                }

                if (dist_r < 362) {
                    img_r.at<uchar>(i, d) *= 1.1;
                    if (img_r.at<uchar>(i, d) >= 255) img_r.at<uchar>(i, d) = 254;
                }
                
            }
        }
        
        vector<Mat> channels;
        channels.push_back(img_b);
        channels.push_back(img_g);
        channels.push_back(img_r);

        merge(channels, outputImg);
        return outputImg;
    }
    
public:
    pipeline (Mat input) {
        // constructor
//        time_t start, end;
//        time(&start);
        this->inputImg = input;
        this->preprocessed = preprocessor(this->inputImg);
        this->processed = k_nearest(this->preprocessed);
        
//        time(&end);
//        double seconds = difftime (end, start);
        
        //        cout << (double)seconds << endl;
    }
    
    Mat visualise () {
        return this->processed;
    }
    
};


int main(int argc, char ** argv) {
    VideoCapture cap(0); // webcam
    Mat source;
    
    while (true) {
        Mat output;
        cap.read(source);
        resize(source, source, Size(), 0.3, 0.3);
        pipeline myPipeline = pipeline(source);
        
        namedWindow("Output", WINDOW_AUTOSIZE);
        
//        resize(myPipeline.visualise(), output, Size(), 2, 2); // upscale it up
        
        imshow("Output", myPipeline.visualise());
        imshow("Source", source);
        waitKey(1);
    }
    return 0;
}
