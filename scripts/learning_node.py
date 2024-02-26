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


from tools.learning_manager import MapLearner
 

import random



hyperparameter_sets = {
    # Scenario 1: Baseline
    'Scenario_1_Moderate': 

    {'alpha': 0.5, 'gamma': 0.9, 't_init': 25, 't_grad': 0.95, 't_min': 0.05, 
     'epsilon_init': 0.5, 'epsilon_grad': 0.97, 'epsilon_min': 0.05},

    # Scenario 2: Faster Learning, High Exploration
    'Scenario_2_Fast_Learning_High_Exploration': 
    {'alpha': 0.7, 'gamma': 0.8, 't_init': 35, 't_grad': 0.93, 't_min': 0.05,
     'epsilon_init': 0.8, 'epsilon_grad': 0.95, 'epsilon_min': 0.1},

    # Scenario 3: Faster Learning, Low Exploration 
    'Scenario_3_Fast_Learning_Low_Exploration': 
    {'alpha': 0.7, 'gamma': 0.85, 't_init': 15, 't_grad': 0.96, 't_min': 0.03,
     'epsilon_init': 0.4, 'epsilon_grad': 0.98, 'epsilon_min': 0.08},

   # Scenario 4: Slower Learning, High Exploration 
   'Scenario_4_Slow_Learning_High_Exploration': 
    {'alpha': 0.3, 'gamma': 0.95, 't_init': 30, 't_grad': 0.99, 't_min': 0.05,
     'epsilon_init': 0.9, 'epsilon_grad': 0.995, 'epsilon_min': 0.2},

    # Scenario 5: Slow Learning, Low Exploration 
    'Scenario_5_Slow_Learning_Low_Exploration': 
    {'alpha': 0.2, 'gamma': 0.7, 't_init': 10, 't_grad': 0.92, 't_min': 0.01,
     'epsilon_init': 0.4, 'epsilon_grad': 0.98, 'epsilon_min': 0.05},

}

exploration_selection = 1  # 1 - Softmax , 2 - Epsilon greedy 
experiment_name= 'softmax_random_start' 



if __name__ == '__main__':
    try:
        rospy.init_node('learning_node', anonymous=True)  # Move ROS node initialization outside the loop
        rate = rospy.Rate(10)

        experiment_log_path = f"/home/ozkan/Desktop/rl_ws/src/scripts/Data/{experiment_name}/"
        for key in hyperparameter_sets:
            print(f'Training with : {key}')
            print(f'Hyperparameters : {hyperparameter_sets[key]}')
            experiment_dir = f"{experiment_log_path}/{key}"
            ML=MapLearner(experiment_log_path=experiment_dir,exploration_selection = exploration_selection,hyperparams_dict=hyperparameter_sets[key])
            ML.run_learner()

        rospy.signal_shutdown('End of hyperparameter search')
   
    except rospy.ROSInterruptException:
        print('Simulation interrupted by the user')
        pass




                            
             


                            














