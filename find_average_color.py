import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import itemfreq
import sys



img = cv2.imread(sys.argv[1])

average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]

print ("\n\n===========================BGR===========================\n")
print ("[B,G,R]: " + str(average_color))


print ("\n\n")

## LAB conversion 
lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
average_color_lab = [lab_image[:, :, i].mean() for i in range(lab_image.shape[-1])]

print ("===========================LAB===========================\n")
print ("Warning: Copy these values straight, dont have to transpose them by dividing by 100 or add 127")
print ("[L,A,B]: " + str(average_color_lab))
