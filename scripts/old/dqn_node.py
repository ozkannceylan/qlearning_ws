#! /usr/bin/env python

import rospy
from time import time
from time import sleep
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState

from sensor_msgs.msg import LaserScan
from math import *
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import yaml
import sys
import os
DATA_PATH = '/home/ozkan/Desktop/rl_ws/src/scripts/Data'
MODULES_PATH = '/home/ozkan/Desktop/rl_ws/src/scripts/'
sys.path.insert(0, MODULES_PATH)



from tools.dqn import MapLearner

import random




if __name__ == '__main__':
    try:
        ML=MapLearner()
        ML.run_learner()
    except rospy.ROSInterruptException:
        print('Simulation interrupted by the user')
        pass





                            
             


                            














