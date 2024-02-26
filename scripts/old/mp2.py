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
from tools.controller import Control
import sys
import yaml
from sensor_msgs.msg import LaserScan


class MotionPlanner(Control):
    def __init__(self,params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml"):
        super().__init__(params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml")
        self.params_yaml_dir = params_yaml_dir
        with open(self.params_yaml_dir, 'r') as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


        self.data_dir= self.params['DATA_PATH']
        self.modules_dir = self.params['MODULES_PATH']
        sys.path.insert(0, self.modules_dir)

        self.q_table_dir = self.params['control_node_params']['q_table_dir']
        self.action_delay = self.params['control_node_params']['action_delay']
        self.init_pos=self.params['control_node_params']['init_positions']
        self.goal_pos=self.params['control_node_params']['goal_positions']
        self.path_mode=self.params['control_node_params']['path_mode_selection']

        self.x_init = self.init_pos['x'][self.path_mode]
        self.y_init = self.init_pos['y'][self.path_mode]
        self.theta_init = self.init_pos['theta'][self.path_mode]

        self.x_goal = self.goal_pos['x'][self.path_mode]
        self.y_goal = self.goal_pos['y'][self.path_mode]
        self.theta_goal = self.goal_pos['theta'][self.path_mode]

       
        self.text = ''
        self.status = ''


    def run_initialize(self):

        self.actions=self.createActions()
        self.state_space=self.createStateSpace()
        self.Q_table=self.readQTable(self.q_table_dir)
        print('Initial Q-table:')
        print(self.Q_table)
        # Init time
        self.t_0 = rospy.Time.now()
        self.t_start = rospy.Time.now()

        # init timer
        while not (self.t_start > self.t_0):
            self.t_start = rospy.Time.now()

        self.t_step = self.t_start
        self.count = 0

        # robot in initial position
        self.robot_in_pos = False

        # because of the video recording
        sleep(1)


    def check_robot_in_pos(self):
        if abs(self.x-self.x_in) < 0.05 and abs(self.y-self.y_in) < 0.05 and abs(self.theta-self.theta_in) < 2:
            self.robot_in_pos = True
            print('\r\nInitial position:')
            print('x = %.2f [m]' % self.x)
            print('y = %.2f [m]' % self.y)
            print('theta = %.2f [degrees]' % self.theta)
            print('')
            sleep(1)
        else:
            self.robot_in_pos = False

    def init_robot_pos(self):
        # Position Initializing
        ( self.x_in , self.y_in , self.theta_in ) = self.robotSetPos(self.setPosPub, self.x_init, self.y_init, self.theta_init)
        # check init pos
        self.odomMsg = rospy.wait_for_message('/odom', Odometry)
        self.get_odom_params()
        print(self.theta, self.theta_init)
        self.check_robot_in_pos()

    def get_odom_params(self):
        (self.x , self.y ) = self.getPosition(self.odomMsg)
        self.theta = (self.getRotation(self.odomMsg))
        return (self.x , self.y , self.theta)

    def get_lidar_params(self):
        ( self.lidar , self.angles ) = self.lidarScan(self.msgScan)
        ( self.state_ind , self.x1 , self.x2 , self.x3 , self.x4 ) = self.scanDiscretization(self.state_space, self.lidar)
    def check_status(self):
        self.crash = self.checkCrash(self.lidar)
        self.object_nearby = self.checkObjectNearby(self.lidar)
        self.goal_near = self.checkGoalNear(self.x, self.y,self.x_goal, self.y_goal)
        self.enable_feedback_control = True

    def crash_status(self): 
        self.robotStop(self.velPub)
        rospy.signal_shutdown('End of testing!')
        self.text = self.text + ' ==> Crash! End of simulation!'
        self.status = 'Crash! End of simulation!'

    def feedback_contol_status(self):

        self.status=self.robotFeedbackControl(self.velPub, self.x, self.y, self.theta, self.x_goal, self.y_goal, radians(self.theta_goal))
        self.text = self.text + ' ==> Feedback Control'
        if self.goal_near:
            self.text = self.text + ' (goal near)'
    def q_learning_status(self):
        (self.action,self.status) = self.getBestAction(self.Q_table, self.state_ind, self.actions)
        if not self.status == 'getBestAction => OK':
            print('\r\n',self.status,'\r\n')
        self.status = self.robotDoAction(self.velPub, self.action)
        if not self.status == 'robotDoAction => OK':
            print('\r\n',self.status,'\r\n')
        self.text = self.text + ' ==> Q-Learning'

    def robot_move_to_goal(self):
        self.count = self.count + 1
        self.text=f"Step: {self.count}, Step time: {self.step_time:.2f} s"
        self.get_odom_params()
        self.get_lidar_params()
        self.check_status()

        if self.crash:  #Crash
            self.crash_status()
        elif self.enable_feedback_control and ( not self.object_nearby or self.goal_near ): #Feedback Control
            self.feedback_contol_status()
        else: #Q-Learning
            self.q_learning_status()
        self.update_text()
        if self.status == 'Goal position reached!':
            self.end_sim()  
            self.text = self.text + '\r\n\r\nGoal position reached! End of simulation!'
        print(self.text)


    def update_text(self):
        self.text += '\r\nCurrent Position:'
        self.text += '\r\n  x:      {:>6.2f} m  (Goal: {:>6.2f} m)'.format(self.x, self.x_goal)
        self.text += '\r\n  y:      {:>6.2f} m  (Goal: {:>6.2f} m)'.format(self.y, self.y_goal)
        self.text += '\r\n  theta:  {:>6.2f} degrees  (Goal: {:>6.2f} degrees)'.format(degrees(self.theta), self.theta_goal)
    def end_sim(self):
        self.robotStop(self.velPub)
        rospy.signal_shutdown('End of testing!')

    def run(self):
        rospy.init_node('control_node', anonymous = False)
        rate = rospy.Rate(10)
        self.setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        self.run_initialize()

        # main loop
        while not rospy.is_shutdown():
            self.msgScan = rospy.wait_for_message('/scan', LaserScan)
            self.odomMsg = rospy.wait_for_message('/odom', Odometry)

            # Secure the minimum time interval between 2 actions
            self.step_time = (rospy.Time.now() - self.t_step).to_sec()

            if self.step_time > self.action_delay:
                print('hi')
                self.t_step = rospy.Time.now()

                if not self.robot_in_pos: # robot not in initial position
                    self.robotStop(self.velPub)
                    self.init_robot_pos()
                else: # robot in initial position => start moving
                    self.robot_move_to_goal()

            
           