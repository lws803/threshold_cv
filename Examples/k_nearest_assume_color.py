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

    def __preprocess(self):
        # Chaining the preprocessors
        self.img = cv2.GaussianBlur(self.img,(5,5),0)
        self.img = norm_illum_color(self.img, 0.8)
        src = self.img
        src_flatten = np.reshape(np.ravel(src, 'C'), (-1, 3))
        dst = np.zeros(src.shape, np.float32)

        colors = np.array([[0x00, 0x00, 0x00],
                           [0xff, 0xff, 0xff],
                           [0xff, 0x00, 0x00],
                           [0x00, 0xff, 0x00],
                           [0x00, 0x00, 0xff]], dtype=np.float32)

        classes = np.array([[0], [1], [2], [3], [4]], np.float32)

        # knn = cv2.ml.KNearest_create()
        # knn.train(colors, classes)
        # retval, result, neighbors, dist = knn.find_nearest(src_flatten.astype(np.float32), 1)

        # dst = colors[np.ravel(result, 'C').astype(np.uint8)]
        # dst = dst.reshape(src.shape).astype(np.uint8)

        diff = ((src[:,:,:,None] - colors.T)**2).sum(axis=2)
        index = diff.argmin(axis=2)
        out = colors[index]

        return out


    def visualisation(self):
        output = self.__preprocess()
        return output




def callback(x):
    pass

# main
if __name__ == '__main__':
    
    print (sys.argv[1])
    img = cv2.imread(sys.argv[1])

    while(True):
        frame = img


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

