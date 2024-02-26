#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

import yaml


class Lidar():
    def __init__(self,params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml"):
        

        self.params_yaml_dir = params_yaml_dir
        with open(self.params_yaml_dir, 'r') as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.MAX_LIDAR_DISTANCE = self.params['lidar_params']['MAX_LIDAR_DISTANCE']
        self.COLLISION_DISTANCE = self.params['lidar_params']['COLLISION_DISTANCE']
        self.NEARBY_DISTANCE = self.params['lidar_params']['NEARBY_DISTANCE']
        self.ZONE_0_LENGTH = self.params['lidar_params']['ZONE_0_LENGTH']
        self.ZONE_1_LENGTH = self.params['lidar_params']['ZONE_1_LENGTH']
        self.ANGLE_MAX = self.params['lidar_params']['ANGLE_MAX']
        self.ANGLE_MIN = self.params['lidar_params']['ANGLE_MIN']
        self.HORIZON_WIDTH = self.params['lidar_params']['HORIZON_WIDTH']



    # Convert LasecScan msg to array
    def lidarScan(self,msgScan):
        distances = np.array([])
        angles = np.array([])

        for i in range(len(msgScan.ranges)):
            angle = degrees(i * msgScan.angle_increment)
            if ( msgScan.ranges[i] > self.MAX_LIDAR_DISTANCE ):
                distance = self.MAX_LIDAR_DISTANCE
            elif ( msgScan.ranges[i] < msgScan.range_min ):
                distance = msgScan.range_min
                # For real robot - protection
                if msgScan.ranges[i] < 0.01:
                    distance = self.MAX_LIDAR_DISTANCE
            else:
                distance = msgScan.ranges[i]

            distances = np.append(distances, distance)
            angles = np.append(angles, angle)

    # distances in [m], angles in [degrees]
        return ( distances, angles )
    def x1_search(self,lidar):
        # Find the left side lidar values of the vehicle
        x1=2
        lidar_left = min(lidar[(self.ANGLE_MIN):(self.ANGLE_MIN + self.HORIZON_WIDTH)])
        if self.ZONE_1_LENGTH > lidar_left > self.ZONE_0_LENGTH:
            x1 = 1 # zone 1
        elif lidar_left <= self.ZONE_0_LENGTH:
            x1 = 0 # zone 0
        return x1
    def x2_search(self,lidar):
        # Find the right side lidar values of the vehicle
        x2=2
        lidar_right = min(lidar[(self.ANGLE_MAX - self.HORIZON_WIDTH):(self.ANGLE_MAX)])
        if self.ZONE_1_LENGTH > lidar_right > self.ZONE_0_LENGTH:
            x2 = 1 # zone 1
        elif lidar_right <= self.ZONE_0_LENGTH:
            x2 = 0 # zone 0
        return x2
    def objectDetection(self,lidar): 

        object = {
            'front': False,
            'left': False,
            'right': False,
            'far_left': False,
            'far_right': False
        }
        # Detection of object in front of the robot
        if ( min(lidar[(self.ANGLE_MAX - self.HORIZON_WIDTH // 3):(self.ANGLE_MAX)]) < 1.0 ) or ( min(lidar[(self.ANGLE_MIN):(self.ANGLE_MIN + self.HORIZON_WIDTH // 3)]) < 1.0 ):
            object['front'] = True
        else:
            object['front']= False

        # Detection of object on the left side of the robot
        if min(lidar[(self.ANGLE_MIN):(self.ANGLE_MIN + 2 * self.HORIZON_WIDTH // 3)]) < 1.0:
            object['left'] = True
        else:
            object['left'] = False

        # Detection of object on the right side of the robot
        if min(lidar[(self.ANGLE_MAX - 2 * self.HORIZON_WIDTH // 3):(self.ANGLE_MAX)]) < 1.0:
            object['right'] = True
        else:
            object['right'] = False

        # Detection of object on the far left side of the robot
        if min(lidar[(self.ANGLE_MIN + self.HORIZON_WIDTH // 3):(self.ANGLE_MIN + self.HORIZON_WIDTH)]) < 1.0:
            object['far_left'] = True
        else:
            object['far_left'] = False

        # Detection of object on the far right side of the robot
        if min(lidar[(self.ANGLE_MAX - self.HORIZON_WIDTH):(self.ANGLE_MAX - self.HORIZON_WIDTH // 3)]) < 1.0:
            object['far_right'] = True
        else:
            object['far_right'] = False

        return object
    
    def x3_search(self,lidar):
        x3=3
        object=self.objectDetection(lidar)

        if (object['front'] and object['left']) and (not object['far_left']):
            x3 = 0 # sector 0
        elif (object['left'] and object['far_left']) and (not object['front']):
            x3 = 1 # sector 1
        elif object['front'] and object['left'] and object['far_left']:
            x3 = 2 # sector 2
        return x3
    def x4_search(self,lidar):
        x4=3
        object=self.objectDetection(lidar)
        if (object['front'] and object['right']) and (not object['far_right']):
            x4 = 0 # sector 0
        elif (object['right'] and object['far_right']) and (not object['front']):
            x4 = 1 # sector 1               
        elif object['front'] and object['right'] and object['far_right']:
            x4 = 2 # sector 2
        return x4
    
    # Discretization of lidar scan
    def scanDiscretization(self,state_space, lidar):

        x1 = self.x1_search(lidar) # Left zone
        x2 = self.x2_search(lidar) # Right zone
        x3 = self.x3_search(lidar) # Left sector
        x4 = self.x4_search(lidar) # Right sector  

        # Find the state space index of (x1,x2,x3,x4) in Q table
        ss = np.where(np.all(state_space == np.array([x1,x2,x3,x4]), axis = 1))
        state_ind = int(ss[0])

        return ( state_ind, x1, x2, x3 , x4 )

    # Check - crash
    def checkCrash(self,lidar):
        lidar_horizon = np.concatenate((lidar[(self.ANGLE_MIN + self.HORIZON_WIDTH):(self.ANGLE_MIN):-1],lidar[(self.ANGLE_MAX):(self.ANGLE_MAX - self.HORIZON_WIDTH):-1]))
        W = np.linspace(1.2, 1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1, 1.2, len(lidar_horizon) // 2))
        if np.min( W * lidar_horizon ) < self.COLLISION_DISTANCE:
            return True
        else:
            return False

    # Check - object nearby
    def checkObjectNearby(self,lidar):
        lidar_horizon = np.concatenate((lidar[(self.ANGLE_MIN + self.HORIZON_WIDTH):(self.ANGLE_MIN):-1],lidar[(self.ANGLE_MAX):(self.ANGLE_MAX - self.HORIZON_WIDTH):-1]))
        W = np.linspace(1.4, 1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1, 1.4, len(lidar_horizon) // 2))
        if np.min( W * lidar_horizon ) < self.NEARBY_DISTANCE:
            print('object nearby')
            return True
            
        else:

            return False

    # Check - goal near
    def checkGoalNear(self,x, y, x_goal, y_goal):
        ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
        if ro < 0.3:
            return True
        else:
            return False