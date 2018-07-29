import cv2
import numpy as np
import math
import requests
import io
from utils.algorithms import norm_illum_color, MovingMedian, shadegrey_lab, power_law, get_salient
import sys

# Usage
# python master_threshold.py <FILENAME>
# adjust slider in treshold bar to tune thresholding 
# Feel free to add more algorithms to the pipeline to suit your needs!
# Press 'q' to end the program and print out thresholding values

class pipeline:
    def __init__ (self, img, thresholds):
        self.img = img
        self.thresholds = thresholds

    def preprocess(self):
        # Chaining the preprocessors
        self.img = cv2.GaussianBlur(self.img,(5,5),0)
        self.img = norm_illum_color(self.img, 0.8)

    
    def thresholding(self):
        l, a, b = cv2.split(cv2.cvtColor(self.img,cv2.COLOR_BGR2LAB))
        mask = get_salient(255-b)
        mask = cv2.threshold(mask, self.thresholds['saliency'] , 255, cv2.THRESH_BINARY)[1]
        
        mask = cv2.dilate(mask,None,iterations=1)
        mask = cv2.erode(mask,None,iterations=1)
        return mask

    def contouring(self, mask):
        i,contours,hierarchy = cv2.findContours(mask ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x)) # Sort the contours 
        return contours


    def blob_detector(self, im):
        min_threshold = 10                      # these values are used to filter our detector.
        max_threshold = 5000                     # they can be tweaked depending on the camera distance, camera angle, ...
        min_area = 50                          # ... focus, brightness, etc.
        min_circularity = .3
        min_inertia_ratio = .5

        params = cv2.SimpleBlobDetector_Params()                # declare filter parameters.
        params.filterByArea = True
        params.filterByCircularity = True
        params.filterByInertia = True
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.minArea = min_area
        params.minCircularity = min_circularity
        params.minInertiaRatio = min_inertia_ratio
        detector = cv2.SimpleBlobDetector_create(params)        # create a blob detector object.
    
        keypoints = detector.detect(im)                         # keypoints is a list containing the detected blobs.
        
        # here we draw keypoints on the frame.
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return im_with_keypoints, keypoints


    def visualisation(self):
        mask = self.thresholding()
        contours = self.contouring(mask)

        mask_bit = cv2.bitwise_and(frame, frame, mask = mask)
        mask_bit = cv2.drawContours(mask_bit ,contours,-1,(255,255,0),1)
        mask_bit, keypoints = self.blob_detector(mask_bit)

        return mask_bit


# Globals
cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)


# Declare more thresholds here, make sure to add them to the dictionary 
thresholds = {'saliency' : 20}

def callback(x):
    pass

cv2.createTrackbar('Saliency', 'threshold', thresholds['saliency'] , 254, callback)

# main
if __name__ == '__main__':
    global thresholds
    
    print (sys.argv[1])
    img = cv2.imread(sys.argv[1])

    while(True):
        frame = img
        thresholds['saliency'] = cv2.getTrackbarPos('Saliency', 'threshold')

        # initialize
        frame_size = frame.shape
        frame_width  = frame_size[1]
        frame_height = frame_size[0]

        frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) # Scale resizing

        my_pipeline = pipeline(frame, thresholds)
        visualisation = my_pipeline.visualisation()

        numpy_horizontal_concat = np.concatenate((frame, visualisation), 1)

        cv2.imshow('image', numpy_horizontal_concat)

        cv2.waitKey(1)
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print ("============= Last recorded values =============")
            print("Threshold val: " + str(thresholds['saliency']))
            break

    cv2.destroyAllWindows()
