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
#include <map>
#include <unordered_map>
#include <dynamic_reconfigure/server.h>
#include <k_segment/k_segmentConfig.h>


#define INT_INF 2147483640

using namespace cv;
using namespace std;
using namespace ros;

image_transport::Publisher image_pub;


class Colors {
    unordered_map<string, vector<float> > colors;
    unordered_map<string, vector<float> > representativeColors;
    unordered_map<string, float> thresholds;
    vector<string> names = {"red", "yellow", "green", "blue"};
    
public:
    Colors () {
        colors["red"] = {138.4, 207.81, 196.89};
        colors["yellow"] = {175.14135491, 116.86378348, 180.24186594};
        colors["blue"] = {0,0,0};
        colors["green"] = {0,0,0};
        
        representativeColors["red"] = {0, 0, 255};
        representativeColors["yellow"] = {0, 255, 255};
        representativeColors["blue"] = {255, 0, 0};
        representativeColors["green"] = {0, 255, 0};

        
        thresholds["red"] = 50;
        thresholds["yellow"] = 20;
        thresholds["blue"] = 0;
        thresholds["green"] = 0;
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
    void setColor (string colorName, vector<float> config, float cutoff) {
    	colors[colorName] = config;
    	thresholds[colorName] = cutoff;
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
        this->inputImg = input;
        this->preprocessed = preprocessor(this->inputImg);
        this->processed = k_nearest(this->preprocessed);
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

                // Output modified video stream
                sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", myPipeline.visualise()).toImageMsg();
                image_pub_.publish(output_msg);

            } else {
                sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
                image_pub_.publish(output_msg);
            }
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
};

void callback(k_segment::k_segmentConfig &config, uint32_t level) {
    vector<float> red = {config.L_red, config.A_red, config.B_red};
    vector<float> yellow = {config.L_yellow, config.A_yellow, config.B_yellow};
    vector<float> green = {config.L_green, config.A_green, config.B_green};
    vector<float> blue = {config.L_blue, config.A_blue, config.B_blue};

    colorBank.setColor("red", red, config.distance_cutoff_red);
    colorBank.setColor("yellow", yellow, config.distance_cutoff_yellow);
    colorBank.setColor("blue", blue, config.distance_cutoff_blue);
    colorBank.setColor("green", green, config.distance_cutoff_green);

}

int main(int argc, char** argv)
{
    init(argc, argv, "k_segment_detector");
    dynamic_reconfigure::Server<k_segment::k_segmentConfig> server;
    dynamic_reconfigure::Server<k_segment::k_segmentConfig>::CallbackType f;

    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);

    ImageConverter ic;

    spin();
    return 0;
}
