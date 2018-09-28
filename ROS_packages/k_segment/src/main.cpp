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

#define INT_INF 2147483640

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
float DISTANCE_LIMIT_FILTER_MANUAL;

bool MANUAL_COLORS_BOOL = false;
float MANUAL_L = 0;
float MANUAL_A = 0;
float MANUAL_B = 0;

float CLIMB = 0.1;


class Colors {
    vector<vector<float> > colors;
    
public:
    Colors () {
        colors = {
            {138.4, 207.81, 196.89}, // Red
            {248.9, 111.25, 220.39} // Yellow
            
        };
    }
    int getNumColors () {
        return colors.size();
    }
    vector<float> getColorFromIndex (int index) {
        return colors[index];
    }
};



class Pipeline {
    Mat inputImg, processed, preprocessed;
    
    float REAL_MIN = INT_INF;
    Colors colorBank;
    
    
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
                
                map<float, vector<float>> colorMap;
                
                for (int c = 0; c < colorBank.getNumColors(); c++) {
                    vector<float> color = colorBank.getColorFromIndex(c);
                    float dist = cartesian_dist(color, lab_channels);
                    if (dist < 60) {
                        colorMap[dist] = color;
                    }
                }
                
                Vec3b currColor;
                
                if (colorMap.empty()) {
                    currColor[0] = 0;
                    currColor[1] = 0;
                    currColor[2] = 0;
                } else {
                    currColor[0] = round(colorMap.begin()->second[0]);
                    currColor[1] = round(colorMap.begin()->second[1]);
                    currColor[2] = round(colorMap.begin()->second[2]);
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


class ImageConverter {
    NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

public:
    ImageConverter()
    : it_(nh_) {
        // Subscrive to input video feed and publish output video feed
        image_sub_ = it_.subscribe("asv/camera2/image_color", 1,
        &ImageConverter::imageCb, this);
        image_pub_ = it_.advertise("k_segment_viewer", 1);
    }

    ~ImageConverter()
    {
    }
    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            if (image_pub_.getNumSubscribers()) {
                resize (cv_ptr->image, cv_ptr->image, Size(), 0.25, 0.25);
                Pipeline myPipeline = Pipeline(cv_ptr->image);
                // cout << myPipeline.getRealMin() << endl;
                MULTIPLER_GLOBAL = myPipeline.getProposedMultipler();

                // Output modified video stream
                sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(std_msgs::Header(), "8UC3", myPipeline.visualise()).toImageMsg();
                image_pub_.publish(output_msg);

            } else {
                sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(std_msgs::Header(), "BGR8", cv_ptr->image).toImageMsg();
                image_pub_.publish(output_msg);
            }
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
};



void callback(k_nearest::k_nearestConfig &config, uint32_t level) {

    DISTANCE_DIFFERENCE_MANUAL_BOOL = config.distance_difference_manual_mode;
    DISTANCE_DIFFERENCE_MANUAL = config.distance_difference;

    DISTANCE_LIMIT_FILTER_MANUAL_BOOL = config.distance_limit_filter_manual_mode;
    DISTANCE_LIMIT_FILTER_MANUAL = (float) config.distance_limit_filter;

    if (DISTANCE_DIFFERENCE_MANUAL_BOOL) {
        ROS_INFO("Setting distance difference: %d", 
            config.distance_difference);

        ROS_INFO("Setting distance limit filter: %lf", 
            config.distance_limit_filter);
    }

    MANUAL_COLORS_BOOL = config.manual_color_set;

    if (MANUAL_COLORS_BOOL) {
        MANUAL_L = config.L;
        MANUAL_A = config.A;
        MANUAL_B = config.B;
    } else {
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
            case 3:
                COLOR_SELECT = "WEIRD_RED";
                break;
            case 4:
                COLOR_SELECT = "WEIRD_GREEN";
                break;
            case 5: 
                COLOR_SELECT = "WEIRD_BLUE";
                break;
            case 6:
                COLOR_SELECT = "WEIRD_YELLOW";
                break;
            case 7:
                COLOR_SELECT = "BRIGHTER_BLUE";
                break;
            case 8:
                COLOR_SELECT = "PURE_YELLOW";
                break;
        }
        ROS_INFO("Setting detection to detect: %s", COLOR_SELECT.c_str());
    }

    CLIMB = config.speed_of_adjustment;
    ROS_INFO("Setting adjustment speed: %lf", CLIMB);
}

int main(int argc, char** argv)
{
    if (argc > 1) {
        vector<string> args(argv, argv + argc);
        cout << args[1] << endl;
        COLOR_SELECT = args[1];
    }
    init(argc, argv, "k_segment_detector");
    ImageConverter ic;

    dynamic_reconfigure::Server<k_nearest::k_nearestConfig> server;
    dynamic_reconfigure::Server<k_nearest::k_nearestConfig>::CallbackType f;

    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);


    spin();
    return 0;
}
