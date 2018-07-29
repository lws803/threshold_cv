#!/usr/bin/env python
import cv2
import numpy as np
from enum import Enum
import rospy

class Color(Enum):
    '''BGR COLOR CODES'''
    black = [0, [0, 0, 0]]
    red = [1, [0, 0, 255]]
    blue = [2, [255, 128, 0]]
    yellow = [3, [0, 255, 255]]
    green = [4, [0, 255, 0]]
    orange = [5, [0, 69, 255]]
    purple = [6, [204, 0, 204]]
    white = [7, [255, 255, 255]]
    cyan = [8, [255, 255, 0]]

    def index(self):
        return self.value[0]

    def bgr(self):
        return self.value[1]

    @staticmethod
    def get_color(index):
        for x in Color:
            if x.index() == index:
                return x
        return None

class Sensor(Enum):
    bottom = 0
    front = 1
    sonar = 2


class Coords(Enum):
    xy = 1
    xyz = 2
    rt = 3

class Object(Enum):
    buoy = 0
    bin = 1
    dice = 2

    @staticmethod
    def get_enum(name):
        for x in Object:
            if x.name == name:
                return x
        return None

