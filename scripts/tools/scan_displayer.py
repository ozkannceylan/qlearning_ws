#! /usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from math import *
from sensor_msgs.msg import LaserScan

import sys
sys.path.insert(0, '/home/ozkan/Desktop/catkin_ws/src/scripts')


from tools.q_learning import QLearning
import yaml



class ScanDisplayer(QLearning):
    def __init__(self,params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml"):
        super().__init__(params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml")
        self.params_yaml_dir = params_yaml_dir
        with open(self.params_yaml_dir, 'r') as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            
        self.MIN_TIME_BETWEEN_SCANS = self.params["scan_node_params"]['MIN_TIME_BETWEEN_SCANS']
        self.MAX_SIMULATION_TIME = self.params["scan_node_params"]['MAX_SIMULATION_TIME']
        if self.MAX_SIMULATION_TIME =="inf":
            self.MAX_SIMULATION_TIME = float('inf')

        self.state_space = self.createStateSpace()

    def init_scanner(self):
        self.init_node()
        self.init_timer()
        self.init_figure()
        
    
    def init_node(self):
        rospy.init_node('scan_node', anonymous = False)
        self.rate = rospy.Rate(10)

        now = datetime.now()
        self.dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
        print('SCAN NODE START ==> ', self.dt_string_start ,'\r\n')

        self.scan_time = 0
        self.count = 0

        self.t_0 = rospy.Time.now()
        self.t_start = rospy.Time.now()
    
    def init_timer(self):
        # init timer
        while not (self.t_start > self.t_0):
            self.t_start = rospy.Time.now()

        self.t = self.t_start
    
    def init_figure(self):
        # Init figure - real time
        plt.style.use('seaborn-ticks')
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1,1,1)
    
    def init_horizon(self):
        self.lidar_horizon = np.concatenate((self.lidar[(self.ANGLE_MIN + self.HORIZON_WIDTH):(self.ANGLE_MIN):-1],self.lidar[(self.ANGLE_MAX):(self.ANGLE_MAX - self.HORIZON_WIDTH):-1]))
        self.angles_horizon = np.linspace(90+self.HORIZON_WIDTH, 90-self.HORIZON_WIDTH, 150)

        self.x_horizon = np.array([])
        self.y_horizon = np.array([])

    def update_timer(self):
        self.scan_time = (rospy.Time.now() - self.t).to_sec()
        self.sim_time = (rospy.Time.now() - self.t_start).to_sec()
        self.count = self.count + 1

    def update_t(self):
        print('\r\nScan cycle:', self.count , '\r\nScan time:', self.scan_time, 's')
        print('Simulation time:', self.sim_time, 's')
        self.t = rospy.Time.now()

    def scan_and_check(self):         
        (self.lidar, self.angles ) = self.lidarScan(self.msgScan)
        (self.state_ind, self.x1, self.x2 ,self.x3 ,self.x4 ) = self.scanDiscretization(self.state_space, self.lidar)

        self.crash = self.checkCrash(self.lidar)
        self.object_nearby = self.checkObjectNearby(self.lidar)

        print('state index:', self.state_ind)
        print('x1 x2 x3 x4')
        print(self.x1, '', self.x2, '', self.x3, '', self.x4)
        if self.crash:
            print('CRASH !')
        if self.object_nearby:
            print('OBJECT NEARBY !')

    def update_horizon(self):
        # horizon in x-y plane
        for i in range(len(self.lidar_horizon)):
            self.x_horizon = np.append(self.x_horizon,self.lidar_horizon[i] * np.cos(radians(self.angles_horizon[i])))
            self.y_horizon = np.append(self.y_horizon,self.lidar_horizon[i] * np.sin(radians(self.angles_horizon[i])))
    
    def plot_scan(self):
        self.ax.clear()
        plt.xlabel('distance[m]')
        plt.ylabel('distance[m]')
        plt.xlim((-1.0,1.0))
        plt.ylim((-0.2,1.2))
        plt.title('Lidar horizon')
        plt.axis('equal')
        self.ax.plot(self.x_horizon, self.y_horizon, 'b.', markersize = 8, label = 'obstacles') 
        self.ax.plot(0, 0, 'r^', markersize = 20, label = 'robot')
        plt.legend(loc = 'lower right', shadow = True)
        plt.draw()
        plt.pause(0.0001)

    def display_scan(self):
        self.update_t()
        self.scan_and_check()
        self.init_horizon()
        self.update_horizon()
        self.plot_scan()

    def end_scan(self):
        now = datetime.now()
        self.dt_string_stop = now.strftime("%d/%m/%Y %H:%M:%S")
        print('\r\nSCAN NODE START ==> ', self.dt_string_start ,'\r\n')
        print('SCAN NODE STOP ==> ', self.dt_string_stop ,'\r\n')
        rospy.signal_shutdown('End of simulation')

    
    def run_scanner(self):
        self.init_scanner()
        # main loop
        while not rospy.is_shutdown():

            self.msgScan = rospy.wait_for_message('/scan', LaserScan)

            self.update_timer()

            if self.scan_time > self.MIN_TIME_BETWEEN_SCANS:
                self.display_scan()

            if self.sim_time > self.MAX_SIMULATION_TIME:
                self.end_scan()
            self.rate.sleep()




if __name__ == '__main__':
    
    scan_displayer = ScanDisplayer()
    try:
        scan_displayer.run_scanner()


    except rospy.ROSInterruptException:
        scan_displayer.end_scan()

        pass
