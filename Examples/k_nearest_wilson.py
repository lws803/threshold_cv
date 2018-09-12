import cv2
import numpy as np
import math
import requests
import io
from utils.algorithms import norm_illum_color, MovingMedian, shadegrey_lab, power_law, get_salient
import sys

multiplier = 10

class pipeline:
    def __init__ (self, img, thresholds):
        self.img = img
        self.thresholds = thresholds
        # self.multiplier = 10

    def __preprocess(self):
        # Chaining the preprocessors
        global multiplier


        img = self.img
        img = cv2.GaussianBlur(self.img,(5,5),0)
        height, width, channels = img.shape

        # img = norm_illum_color(img, 0.8)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(img)
        # color = np.array((54.29/100 * 255, 80.81 + 127, 69.89 + 127)) # Red
        color = np.array((87.82/100 * 255, -79.29 + 127,  80.99 + 127)) # Green


        # TODO: Add trackbar to control the thresholding
        
        blank_image = np.zeros((height,width,1), np.uint8)
        min_dist = 100000
        max_dist = 0

        for y in range(height):
            for x in range(width):
                dist = np.linalg.norm(np.array((a[y][x], b[y][x])) - [color[1], color[2]])

                # dist1 = math.sqrt((a[y][x] - color[1]) ** 2)
                # dist2 = math.sqrt((b[y][x] - color[2]) ** 2)
                # dist = np.amin([dist1, dist2])

                # Scaling

                # TODO: Add an automatic adjustment to adjust when the seperation is too little. (must be at least >= 100)
                # TODO: Add a technique to increase the seperation between the colors.
                dist = dist * multiplier

                min_dist = np.amin([min_dist, dist])
                max_dist = np.amax([max_dist, dist])

                dist = dist - min_dist
                
                if (dist > 255):
                    dist = 255
                
                if (dist < 0): 
                    dist = 0


                blank_image[y][x] = (255 - dist)

        print ("Seperation: ")
        print ((max_dist - min_dist))

        if (max_dist - min_dist < 1000):
            multiplier = multiplier + 5

        return blank_image, min_dist


    def __threshold(self, processed_img, min_dist):
        mask = cv2.inRange(processed_img, 255 - min_dist - 20, 255)

        return mask


    def visualisation(self):
        output, min_dist = self.__preprocess()
        # output = self.__threshold(output, min_dist)
        return output




def callback(x):
    pass

# main
if __name__ == '__main__':
    
    print (sys.argv[1])
    img = cv2.imread(sys.argv[1])
    # cap = cv2.VideoCapture(0)

    while(True):
        frame = img
        # ret, frame = cap.read()


        # initialize
        frame_size = frame.shape
        frame_width  = frame_size[1]
        frame_height = frame_size[0]

        frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) # Scale resizing

        my_pipeline = pipeline(frame, {})
        visualisation = my_pipeline.visualisation()

        # numpy_horizontal_concat = np.concatenate((frame, visualisation), 1)

        cv2.imshow('image', visualisation)

        cv2.waitKey(1)
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

