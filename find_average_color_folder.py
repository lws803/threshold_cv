import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import itemfreq
import sys
import os

# Usage: python find_average_color.py FOLDER_PATH

average_color_lab = [[0,0,0]]
average_color = [[0,0,0]]
average_color_hsv = [[0,0,0]]

numFiles = 0

for r, d, f in os.walk(sys.argv[1]):
    for file in f:
        if not os.path.isfile(file):
            img = cv2.imread(str(sys.argv[1] + "/" + file))
            # BGR conversion
            average_color += [[img[:, :, i].mean() for i in range(img.shape[-1])]]

            ## LAB conversion 
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            average_color_lab += [[lab_image[:, :, i].mean() for i in range(lab_image.shape[-1])]]

            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            average_color_hsv += [[hsv_image[:, :, i].mean() for i in range(hsv_image.shape[-1])]]

            numFiles += 1


sumBGR = np.sum(average_color, axis=0)
sumLAB = np.sum(average_color_lab, axis=0)
sumHSV = np.sum(average_color_hsv, axis=0)

print ("\n\n===========================BGR===========================\n")
print ("[B,G,R]: " + str(np.true_divide(sumBGR, numFiles)))


print ("\n\n")

print ("\n\n===========================HSV===========================\n")
print ("[H,S,V]: " + str(np.true_divide(sumHSV, numFiles)))

print ("\n\n")

print ("===========================LAB===========================\n")
print ("Warning: Copy these values straight, dont have to transpose them by dividing by 100 or add 127")
print ("[L,A,B]: " + str(np.true_divide(sumLAB, numFiles)))
