#!/usr/bin/env python
from std_msgs.msg._Float32 import Float32
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from sensor_msgs.msg import Image, Imu
from vision.srv import Detector
from controls.msg import LocomotionAction
from controls.srv import Controller
from acoustics.msg import Ping
from bbauv_msgs.srv import navigate2d
from sonard.msg import SonarInfo
from nav_msgs.msg import Odometry

class Ros():

    # List of topics
    topics = {'depth': ('/auv/depth', Float32),
              'front_camera': ('/auv/front_cam/image_color', Image),
              'bottom_camera': ('/auv/bot_cam/image_color', Image),
              'sonar': ('/auv/sonar/image', Image),
              'sonar_info': ('/auv/sonar/info', SonarInfo),
              'detector_server': ('/vision/detector', Detector),
              'nav_server': ('LocomotionServer', LocomotionAction),
              'controller': ('/controller', Controller),
              'navigate2D': ('/navigate2D', navigate2d),
              'acoustics': ('/acoustics/ping', Ping),
              'rpy': ('/navigation/RPY', Vector3Stamped),
              'imu_vel': ('/AHRS8/data', Imu),
              'veh_vel': ('/navigation/odom/relative', Odometry),
              'veh_pos': ('/navigation/odom/earth', PoseStamped)}

    # List of reconfigurable nodes
    reconf_nodes = {'bottom_camera': '/auv/bot_cam',
                    'front_camera': '/auv/front_cam'}
