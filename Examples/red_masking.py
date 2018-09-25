import cv2
import numpy as np
import math
import requests
import io
from utils.algorithms import norm_illum_color, MovingMedian, shadegrey_lab, power_law, get_salient
import sys



class pipeline:
    def __init__ (self, img, thresholds):
        self.img = img
        self.thresholds = thresholds

    def preprocess(self):
        # Chaining the preprocessors
        self.img = cv2.GaussianBlur(self.img,(5,5),0)
        # self.img = norm_illum_color(self.img, 0.8)

    
    def thresholding(self):
        # l, a, b = cv2.split(cv2.cvtColor(self.img,cv2.COLOR_BGR2LAB))
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)

        upper = np.array([self.thresholds['L_high'], self.thresholds['A_high'], self.thresholds['B_high']])
        lower = np.array([self.thresholds['L_low'], self.thresholds['A_low'], self.thresholds['B_low']])

        mask = cv2.inRange(lab, lower, upper)

        kernel = np.ones((5,5), np.uint8)

        mask = cv2.dilate(mask,kernel,iterations = 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.erode(mask, None,iterations=4) # Can change iterations here
        
        return mask

    def contouring(self, mask):
        i,contours,hierarchy = cv2.findContours(mask ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x)) # Sort the contours 
        return contours



    def visualisation(self):
        self.preprocess()
        mask = self.thresholding()
        contours = self.contouring(mask)

        output = self.img
        mask_bit = cv2.drawContours(output ,contours,-1,(255,255,0),1)
        output = cv2.bitwise_and(output, output, mask = mask)

        count = 0
        for c in reversed(contours): 
            if (cv2.contourArea(c) > 4):
                rect = cv2.boundingRect(c)
                cv2.rectangle(output,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), (0,255,0),2)
                count += 1
                if (count > 2):
                    break

        return output


# Globals
cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)


# Declare more thresholds here, make sure to add them to the dictionary 
thresholds = {'L_high' : 117, 'A_high': 127, 'B_high': 254, 'L_low': 0, 'A_low': 119, 'B_low': 135}

def callback(x):
    pass

cv2.createTrackbar('L_high', 'threshold', thresholds['L_high'] , 254, callback)
cv2.createTrackbar('A_high', 'threshold', thresholds['A_high'] , 254, callback)
cv2.createTrackbar('B_high', 'threshold', thresholds['B_high'] , 254, callback)
cv2.createTrackbar('L_low', 'threshold', thresholds['L_low'] , 254, callback)
cv2.createTrackbar('A_low', 'threshold', thresholds['A_low'] , 254, callback)
cv2.createTrackbar('B_low', 'threshold', thresholds['B_low'] , 254, callback)

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

