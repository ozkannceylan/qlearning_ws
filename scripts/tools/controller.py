#!/usr/bin/env python

import rospy
from time import time
from time import sleep
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from math import *
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tools.q_learning import QLearning
import sys
import yaml
from sensor_msgs.msg import LaserScan


class Control(QLearning):
    def __init__(self,params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml"):
        super().__init__(params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml")
        self.params_yaml_dir = params_yaml_dir
        with open(self.params_yaml_dir, 'r') as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.CONST_LINEAR_SPEED_FORWARD = self.params['control_parameters']['CONST_LINEAR_SPEED_FORWARD']
        self.CONST_ANGULAR_SPEED_FORWARD = self.params['control_parameters']['CONST_ANGULAR_SPEED_FORWARD']
        self.CONST_LINEAR_SPEED_TURN = self.params['control_parameters']['CONST_LINEAR_SPEED_TURN']
        self.CONST_ANGULAR_SPEED_TURN = self.params['control_parameters']['CONST_ANGULAR_SPEED_TURN']
        
        self.K_RO = self.params['feedback_control_parameters']['K_RO']
        self.K_ALPHA = self.params['feedback_control_parameters']['K_ALPHA']
        self.K_BETA = self.params['feedback_control_parameters']['K_BETA']
        self.V_CONST = self.params['feedback_control_parameters']['V_CONST']
        
        self.GOAL_DIST_THRESHOLD = self.params['goal_reaching_threshold']['GOAL_DIST_THRESHOLD']
        self.GOAL_ANGLE_THRESHOLD = self.params['goal_reaching_threshold']['GOAL_ANGLE_THRESHOLD']



    def getRotation(self,odomMsg):
        orientation_q = odomMsg.pose.pose.orientation
        orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        return yaw
    def getPosition(self,odomMsg):
        x = odomMsg.pose.pose.position.x
        y = odomMsg.pose.pose.position.y
        return ( x , y)
    
    def getLinVel(self,odomMsg):
        return odomMsg.twist.twist.linear.x 
    
    def getAngVel(self,odomMsg):
        return odomMsg.twist.twist.angular.z
    
    def createVelMsg(self,v,w):
        velMsg = Twist()
        velMsg.linear.x = v
        velMsg.linear.y = 0
        velMsg.linear.z = 0
        velMsg.angular.x = 0
        velMsg.angular.y = 0
        velMsg.angular.z = w
        return velMsg
    
    def robotGoForward(self,velPub):
        velMsg = self.createVelMsg(self.CONST_LINEAR_SPEED_FORWARD,self.CONST_ANGULAR_SPEED_FORWARD)
        velPub.publish(velMsg)
    
    def robotTurnLeft(self,velPub):
        velMsg = self.createVelMsg(self.CONST_LINEAR_SPEED_TURN,+self.CONST_ANGULAR_SPEED_TURN)
        velPub.publish(velMsg)
    
    def robotTurnRight(self,velPub):   
        velMsg = self.createVelMsg(self.CONST_LINEAR_SPEED_TURN,-self.CONST_ANGULAR_SPEED_TURN)
        velPub.publish(velMsg)
    
    def robotStop(self,velPub):
        velMsg = self.createVelMsg(0.0,0.0)
        velPub.publish(velMsg)
    
    def robotSetPos(self,setPosPub,x,y,theta):
        checkpoint = ModelState()

        checkpoint.model_name = 'turtlebot3_burger'

        checkpoint.pose.position.x = x
        checkpoint.pose.position.y = y
        checkpoint.pose.position.z = 0.0

        [x_q,y_q,z_q,w_q] = quaternion_from_euler(0.0,0.0,radians(theta))

        checkpoint.pose.orientation.x = x_q
        checkpoint.pose.orientation.y = y_q
        checkpoint.pose.orientation.z = z_q
        checkpoint.pose.orientation.w = w_q

        checkpoint.twist.linear.x = 0.0
        checkpoint.twist.linear.y = 0.0
        checkpoint.twist.linear.z = 0.0

        checkpoint.twist.angular.x = 0.0
        checkpoint.twist.angular.y = 0.0
        checkpoint.twist.angular.z = 0.0

        setPosPub.publish(checkpoint)
        return ( x , y , theta )

    def robotSetRandomPos(self,setPosPub):
        x_range = np.array([-0.4, 0.6, 0.6, -1.4, -1.4, 2.0, 2.0, -2.5, 1.0, -1.0])
        y_range = np.array([-0.4, 0.6, -1.4, 0.6, -1.4, 1.0, -1.0, 0.0, 2.0, 2.0])
        theta_range = np.arange(0, 360, 15)
        #theta_range = np.array([0, 30, 45, 60, 75, 90])

        ind = np.random.randint(0,len(x_range))
        ind_theta = np.random.randint(0,len(theta_range))

        x = x_range[ind]
        y = y_range[ind]
        theta = theta_range[ind_theta]

        checkpoint = ModelState()

        checkpoint.model_name = 'turtlebot3_burger'

        checkpoint.pose.position.x = x
        checkpoint.pose.position.y = y
        checkpoint.pose.position.z = 0.0

        [x_q,y_q,z_q,w_q] = quaternion_from_euler(0.0,0.0,radians(theta))

        checkpoint.pose.orientation.x = x_q
        checkpoint.pose.orientation.y = y_q
        checkpoint.pose.orientation.z = z_q
        checkpoint.pose.orientation.w = w_q

        checkpoint.twist.linear.x = 0.0
        checkpoint.twist.linear.y = 0.0
        checkpoint.twist.linear.z = 0.0

        checkpoint.twist.angular.x = 0.0
        checkpoint.twist.angular.y = 0.0
        checkpoint.twist.angular.z = 0.0

        setPosPub.publish(checkpoint)
        return ( x , y , theta )
    
    def robotDoAction(self,velPub, action):
        status = 'robotDoAction => OK'
        if action == 0:
            self.robotGoForward(velPub)
        elif action == 1:
            self.robotTurnLeft(velPub)
        elif action == 2:
            self.robotTurnRight(velPub)
        else:
            status = 'robotDoAction => INVALID ACTION'
            self.robotGoForward(velPub)

        return status
    
    def robotFeedbackControl(self,velPub, x, y, theta, x_goal, y_goal, theta_goal):
            # theta goal normalization
        if theta_goal >= pi:
            theta_goal_norm = theta_goal - 2 * pi
        else:
            theta_goal_norm = theta_goal

        ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
        lamda = atan2( y_goal - y , x_goal - x )

        alpha = (lamda -  theta + pi) % (2 * pi) - pi
        beta = (theta_goal - lamda + pi) % (2 * pi) - pi

        if ro < self.GOAL_DIST_THRESHOLD and degrees(abs(theta-theta_goal_norm)) < self.GOAL_ANGLE_THRESHOLD:
            status = 'Goal position reached!'
            v = 0
            w = 0
            v_scal = 0
            w_scal = 0
        else:
            status = 'Goal position not reached!'
            v = self.K_RO * ro
            w = self.K_ALPHA * alpha + self.K_BETA * beta
            v_scal = v / abs(v) * self.V_CONST
            w_scal = w / abs(v) * self.V_CONST

        velMsg = self.createVelMsg(v_scal, w_scal)
        velPub.publish(velMsg)

        return status
    
        # Stability Condition
    def check_stability(self,k_rho, k_alpha, k_beta):
        return k_rho > 0 and k_beta < 0 and k_alpha > k_rho

    # Strong Stability Condition
    def check_strong_stability(self,k_rho, k_alpha, k_beta):
        return k_rho > 0 and k_beta < 0 and k_alpha + 5 * k_beta / 3 - 2 * k_rho / np.pi > 0
   