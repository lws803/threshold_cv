import cv2
import numpy as np
import math
import requests
import io
from utils.algorithms import norm_illum_color, MovingMedian, shadegrey_lab, power_law, get_salient, deilluminate, gamma_correct, contrast_stretching, sharpen
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
        img = self.img
        # img = deilluminate(img)
        # img = gamma_correct(img)
        img = norm_illum_color(img, 0.05)
        img = sharpen(img)
        img = cv2.GaussianBlur(img,(5,5),0)

        # img = contrast_stretching(img)
        return img

    
    def thresholding(self):
        lab = cv2.cvtColor(self.preprocess(), cv2.COLOR_BGR2Lab)

        upper = np.array([self.thresholds['L_high'], self.thresholds['A_high'], self.thresholds['B_high']])
        lower = np.array([self.thresholds['L_low'], self.thresholds['A_low'], self.thresholds['B_low']])

        mask = cv2.inRange(lab, lower, upper)
        
        # mask = cv2.dilate(mask,None,iterations=1)
        # mask = cv2.erode(mask,None,iterations=1)
        return mask

    def contouring(self, mask):
        i,contours,hierarchy = cv2.findContours(mask ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x)) # Sort the contours 
        return contours


    def blob_detector(self, im):
        min_threshold = 10                      # these values are used to filter our detector.
        max_threshold = 5000                     # they can be tweaked depending on the camera distance, camera angle, ...
        min_area = 20                          # ... focus, brightness, etc.
        min_circularity = .9
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
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return im_with_keypoints, keypoints


    def cartesian_distance_rect (self, rect, pt):
        midX = (rect[0] + rect[2]/2)
        midY = (rect[1] + rect[3]/2)
        diffX = midX - pt[0]
        diffY = midY - pt[1]

        return math.sqrt(diffX**2 + diffY**2)


    def visualisation(self):
        # self.preprocess()

        mask = self.thresholding()
        contours = self.contouring(mask)

        output = self.preprocess()
        output = cv2.drawContours(output ,contours,-1,(255,255,0),1)
        output = cv2.bitwise_and(output, output, mask = mask)
        output, keypoints = self.blob_detector(output)


        detected = False
        # Find the largest contours 
        # Only mark the contours which contains the blob
        for c in reversed(contours):
            if (cv2.contourArea(c) > 5):
                rect = cv2.boundingRect(c)
                for point in keypoints:

                    if (self.cartesian_distance_rect(rect, point.pt) < rect[2]/2 and self.cartesian_distance_rect(rect, point.pt) < rect[3]/2 and not detected):
                        cv2.rectangle(output,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), (0,255,0),2)
                        detected = True

        return output


# Globals
cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)


# Declare more thresholds here, make sure to add them to the dictionary 
thresholds = {'L_high' : 255, 'A_high': 255, 'B_high': 255, 'L_low': 84, 'A_low': 123, 'B_low': 82}

def callback(x):
    pass

cv2.createTrackbar('L_high', 'threshold', thresholds['L_high'] , 255, callback)
cv2.createTrackbar('A_high', 'threshold', thresholds['A_high'] , 255, callback)
cv2.createTrackbar('B_high', 'threshold', thresholds['B_high'] , 255, callback)
cv2.createTrackbar('L_low', 'threshold', thresholds['L_low'] , 255, callback)
cv2.createTrackbar('A_low', 'threshold', thresholds['A_low'] , 255, callback)
cv2.createTrackbar('B_low', 'threshold', thresholds['B_low'] , 255, callback)


# main
if __name__ == '__main__':
    global thresholds
    
    print (sys.argv[1])
    img = cv2.imread(sys.argv[1])

    while(True):
        frame = img
        thresholds['L_high'] = cv2.getTrackbarPos('L_high', 'threshold')
        thresholds['A_high'] = cv2.getTrackbarPos('A_high', 'threshold')
        thresholds['B_high'] = cv2.getTrackbarPos('B_high', 'threshold')
        thresholds['L_low'] = cv2.getTrackbarPos('L_low', 'threshold')
        thresholds['A_low'] = cv2.getTrackbarPos('A_low', 'threshold')
        thresholds['B_low'] = cv2.getTrackbarPos('B_low', 'threshold')
        
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
            print ("High: [" + str(thresholds['L_high']) + ", " + str(thresholds['A_high']) + ", " + str(thresholds['B_high']) + "]")
            print ("Low: [" + str(thresholds['L_low']) + ", " + str(thresholds['A_low']) + ", " + str(thresholds['B_low']) + "]")
            break

    cv2.destroyAllWindows()
