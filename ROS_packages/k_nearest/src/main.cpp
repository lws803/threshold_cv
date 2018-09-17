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
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "ros/ros.h"
#include <dynamic_reconfigure/server.h>
#include <k_nearest/k_nearestConfig.h>

using namespace cv;
using namespace std;
using namespace ros;

float MULTIPLER_GLOBAL = 10;
float MIN_GLOBAL, MAX_GLOBAL;
image_transport::Publisher image_pub;

string COLOR_SELECT = "PURE_GREEN";
bool DISTANCE_DIFFERENCE_MANUAL_BOOL = false;
int DISTANCE_DIFFERENCE_MANUAL;
bool DISTANCE_LIMIT_FILTER_MANUAL_BOOL = false;
int DISTANCE_LIMIT_FILTER_MANUAL;


class ColorMap {
    vector<float> colorArray = {0,0,0};
    int DISTANCE_DIFFERENCE = 1000;
    float DISTANCE_LIMIT_FILTER = 255; // More means larger allowance

public:
    ColorMap (string COLOR_SELECT) {
        if (COLOR_SELECT == "PURE_RED") {
            colorArray = {54.29/100 * 255, 80.81 + 127, 69.89 + 127};
            DISTANCE_DIFFERENCE = 800;
        }
        else if (COLOR_SELECT == "PURE_GREEN") {
            colorArray = {46.228/100 * 255, -51.699 + 127,  49.897 + 127};
            DISTANCE_DIFFERENCE = 500;
        }
        else if (COLOR_SELECT == "PURE_BLUE") {
            colorArray = {29.57/100 * 255, 68.30 + 127,  -112.03 + 127};
        }
        else if (COLOR_SELECT == "PURE_YELLOW") {
            colorArray = {97.139/100 * 255, -21.558 + 127,  94.477 + 127};
        }
        else if (COLOR_SELECT == "DARK_GREEN") {
            colorArray = {36.202/100 * 255, -43.37 + 127,  41.858 + 127};
        }
        else if (COLOR_SELECT == "WEIRD_GREEN") {
            colorArray = {40.57/100 * 255,  -11.08 + 127, -2.39 + 127};
            DISTANCE_DIFFERENCE = 2000;
        }
        else if (COLOR_SELECT == "WEIRD_RED") {
            colorArray = {50.52/100 * 255,  39.26 + 127, 25.71 + 127};
        }
        else {
            colorArray = {97.139/100 * 255, -21.558 + 127,  94.477 + 127}; // PURE_YELLOW
        }

        if (DISTANCE_DIFFERENCE_MANUAL_BOOL) {
            DISTANCE_DIFFERENCE = DISTANCE_DIFFERENCE_MANUAL;
        }

        if (DISTANCE_LIMIT_FILTER_MANUAL_BOOL) {
            DISTANCE_LIMIT_FILTER = DISTANCE_LIMIT_FILTER_MANUAL;
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
};


class pipeline {
    Mat inputImg, processed, mask, preprocessed;
    ColorMap myColorChoice = ColorMap(COLOR_SELECT);

    float MULTIPLIER = 10;
    
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
        Mat lab_image = input;
        
        // Gaussian blurring
        GaussianBlur(input, lab_image, Size( 5, 5 ), 0, 0);
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
        
        float min = 255 * MULTIPLIER;
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
                    dist = multiply_dist(dist);
                    
                    if (dist < min) min = dist;
                    if (dist > max) max = dist;
                    
                    dist -= MIN_GLOBAL;
                    
                    dist = cut_off_dist(dist);
                }                
                img.at<uchar>(i, d) = 255 - round(dist);
            }
        }
        
        MAX_GLOBAL = max;
        MIN_GLOBAL = min;
        
        autoAdjust(max, min);
        
        return img;
    }
    
    void autoAdjust (float max, float min) {
        if (max - min < myColorChoice.getDistanceDifference()) MULTIPLIER += 0.1;
        if (max - min > myColorChoice.getDistanceDifference()) MULTIPLIER -= 0.1;
        if (MULTIPLIER < 1) MULTIPLIER = 1;
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
};


class ImageConverter {
    NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

public:
    ImageConverter()
    : it_(nh_) {
        // Subscrive to input video feed and publish output video feed
        image_sub_ = it_.subscribe("/asv/camera2/image_color/", 1,
        &ImageConverter::imageCb, this);
        image_pub_ = it_.advertise("k_nearest_viewer", 1);


    }

    ~ImageConverter()
    {
    }
    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            resize (cv_ptr->image, cv_ptr->image, Size(), 0.3, 0.3);
            pipeline myPipeline = pipeline(cv_ptr->image, MULTIPLER_GLOBAL);
            MULTIPLER_GLOBAL = myPipeline.getProposedMultipler();
            sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(std_msgs::Header(), "8UC1", myPipeline.visualise()).toImageMsg();

            // Output modified video stream
            image_pub_.publish(output_msg);

        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
};

void callback(k_nearest::k_nearestConfig &config, uint32_t level) {
    ROS_INFO("Setting distance difference: %d", 
            config.distance_difference);

    ROS_INFO("Setting distance limit filter: %d", 
            config.distance_limit_filter);

    DISTANCE_DIFFERENCE_MANUAL_BOOL = config.distance_difference_manual_mode;
    DISTANCE_DIFFERENCE_MANUAL = config.distance_difference;

    DISTANCE_LIMIT_FILTER_MANUAL_BOOL = config.distance_limit_filter_manual_mode;
    DISTANCE_LIMIT_FILTER_MANUAL = config.distance_limit_filter;

    switch (config.color_selection) {
        case 0:
            COLOR_SELECT = "PURE_RED";
            break;
        case 1: 
            COLOR_SELECT = "PURE_GREEN";
            break;
        case 2: 
            COLOR_SELECT = "PURE_BLUE";
            break;
    }
    ROS_INFO("Setting detection to detect: %s", COLOR_SELECT.c_str());
}

int main(int argc, char** argv)
{
    if (argc > 1) {
        vector<string> args(argv, argv + argc);
        cout << args[1] << endl;
        COLOR_SELECT = args[1];
    }
    init(argc, argv, "k_nearest_detector");
    ImageConverter ic;

    dynamic_reconfigure::Server<k_nearest::k_nearestConfig> server;
    dynamic_reconfigure::Server<k_nearest::k_nearestConfig>::CallbackType f;

    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);


    spin();
    return 0;
}
