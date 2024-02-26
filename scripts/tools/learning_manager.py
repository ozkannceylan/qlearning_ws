#! /usr/bin/env python

import rospy
from time import time
from time import sleep
from datetime import datetime as dt
import pandas as pd
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
MODULES_PATH = '/home/ozkan/Desktop/rl_ws/src/'
sys.path.insert(0, MODULES_PATH)


from tools.controller import Control
from tools.constants import *


class MapLearner(Control):
    def __init__(self,params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml",experiment_log_path="/home/ozkan/Desktop/rl_ws/src/scripts/Data/Data_Logs11/",exploration_selection=1,hyperparams_dict=None):
        super().__init__(params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml")


        self.params_yaml_dir= params_yaml_dir
        with open(params_yaml_dir, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # Episode parameters
        self.MAX_EPISODES = params['episode_parameters']['max_episodes']
        self.MAX_STEPS_PER_EPISODE = params['episode_parameters']['max_steps_per_episode']
        self.MIN_TIME_BETWEEN_ACTIONS = params['episode_parameters']['min_time_between_actions']
        # Learning parameters
        # self.ALPHA=params['learning_parameters']['alpha']
        # self.GAMMA=params['learning_parameters']['gamma']
        # self.T_INIT=params['learning_parameters']['t_init']
        # self.T_GRAD=params['learning_parameters']['t_grad']
        # self.T_MIN=params['learning_parameters']['t_min']
        # self.EPSILON_INIT=params['learning_parameters']['epsilon_init']
        # self.EPSILON_GRAD=params['learning_parameters']['epsilon_grad']
        # self.EPSILON_MIN=params['learning_parameters']['epsilon_min']

        self.ALPHA = hyperparams_dict['alpha']
        self.GAMMA = hyperparams_dict['gamma']
        self.T_INIT = hyperparams_dict['t_init']
        self.T_GRAD = hyperparams_dict['t_grad']
        self.T_MIN = hyperparams_dict['t_min']
        self.EPSILON_INIT = hyperparams_dict['epsilon_init']
        self.EPSILON_GRAD = hyperparams_dict['epsilon_grad']
        self.EPSILON_MIN = hyperparams_dict['epsilon_min']

        # Exploration function
        # self.EXPLORATION_FUNCTION = params['exploration_selection']
        self.EXPLORATION_FUNCTION = exploration_selection

        # Initial position
        self.X_INIT = params['initial_position']['x_init']
        self.Y_INIT = params['initial_position']['y_init']
        self.THETA_INIT = params['initial_position']['theta_init']
        self.RANDOM_INIT_POS = params['initial_position']['random_init_pos']
        # Log file directory
        self.LOG_FILE_DIR = experiment_log_path
        os.makedirs(self.LOG_FILE_DIR, exist_ok=True)
        self.info_logs=params['dir']['info_logs_file']
        # os.makedirs(self.LOG_FILE_DIR+self.info_logs, exist_ok=True)
        self.sim_metrics=params['dir']['sim_metrics_logs_file']
        # rospy.init_node('learning_node', anonymous = False)
        # rate = rospy.Rate(10)

        self.setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
        self.init_Learner()
       
        


    def init_QLearning(self):
        self.actions = self.createActions()
        self.state_space = self.createStateSpace()
        self.Q_table = self.createQTable(len(self.state_space),len(self.actions))
        print('Initial Q-table:')
        print(self.Q_table)

    def open_log_files(self):
        self.log_sim_info = open(self.LOG_FILE_DIR+self.info_logs,'w+')
        self.log_sim_params = open(self.LOG_FILE_DIR+ self.sim_metrics,'w+')



    def init_learning_params(self):
        self.T = self.T_INIT
        self.EPSILON = self.EPSILON_INIT
        self.alpha = self.ALPHA
        self.gamma = self.GAMMA

    def init_episodes_params(self): 
        self.ep_steps = 0
        self.ep_reward = 0
        self.episode = 1
        self.crash = 0
        self.reward_max_per_episode = np.array([])
        self.reward_min_per_episode = np.array([])
        self.reward_avg_per_episode = np.array([])
        self.ep_reward_arr = np.array([])
        self.steps_per_episode = np.array([])
        self.reward_per_episode = np.array([])

    def init_init_pos(self):
        # initial position
        self.robot_in_pos = False
        self.first_action_taken = False


    def initTime(self):
        # init time
        self.t_0 = rospy.Time.now()
        self.t_start = rospy.Time.now()

        # init timer
        while not (self.t_start > self.t_0):
            self.t_start = rospy.Time.now()

        self.t_ep = self.t_start
        self.t_sim_start = self.t_start
        self.t_step = self.t_start

        self.T_per_episode = np.array([])
        self.EPSILON_per_episode = np.array([])
        self.t_per_episode = np.array([])

        self.start_dt = dt.now()


    def init_log_files(self):
        
        self.dt_string_start = dt.now().strftime("%d/%m/%Y %H:%M:%S")
        # Log date to files
        text = '\r\n' + 'SIMULATION START ==> ' + self.dt_string_start + '\r\n\r\n'
        print(text)
        self.log_sim_info.write(text)
        self.log_sim_params.write(text)




    def log_simulation_params(self):
        text_lines = [
            "\nSimulation parameters:",
            "---------------------------------------",
            "INITIAL POSITION = RANDOM" if self.RANDOM_INIT_POS else f"INITIAL POSITION = ( {self.X_INIT:.2f} , {self.Y_INIT:.2f} , {self.THETA_INIT:.2f} )",
            "---------------------------------------",
            f"MAX_EPISODES = {self.MAX_EPISODES}",
            f"MAX_STEPS_PER_EPISODE = {self.MAX_STEPS_PER_EPISODE}",
            f"MIN_TIME_BETWEEN_ACTIONS = {self.MIN_TIME_BETWEEN_ACTIONS:.2f} s",
            "---------------------------------------",
            f"ALPHA = {self.ALPHA:.2f}",
            f"GAMMA = {self.GAMMA:.2f}",
        ]
        if self.EXPLORATION_FUNCTION == 1:
            text_lines.extend([
                f"T_INIT = {self.T_INIT:.3f}",
                f"T_GRAD = {self.T_GRAD:.3f}",
                f"T_MIN = {self.T_MIN:.3f}"
            ])
        else:
            text_lines.extend([
                f"EPSILON_INIT = {self.EPSILON_INIT:.3f}",
                f"EPSILON_GRAD = {self.EPSILON_GRAD:.3f}",
                f"EPSILON_MIN = {self.EPSILON_MIN:.3f}"
            ])

        text_lines.extend([
            "---------------------------------------",
            f"MAX_LIDAR_DISTANCE = {self.MAX_LIDAR_DISTANCE:.2f}",
            f"COLLISION_DISTANCE = {self.COLLISION_DISTANCE:.2f}",
            f"ZONE_0_LENGTH = {self.ZONE_0_LENGTH:.2f}",
            f"ZONE_1_LENGTH = {self.ZONE_1_LENGTH:.2f}",
            "---------------------------------------",
            f"CONST_LINEAR_SPEED_FORWARD = {self.CONST_LINEAR_SPEED_FORWARD:.3f}",
            f"CONST_ANGULAR_SPEED_FORWARD = {self.CONST_ANGULAR_SPEED_FORWARD:.3f}",
            f"CONST_LINEAR_SPEED_TURN = {self.CONST_LINEAR_SPEED_TURN:.3f}",
            f"CONST_ANGULAR_SPEED_TURN = {self.CONST_ANGULAR_SPEED_TURN:.3f}"
        ])

        log_text = '\n'.join(text_lines)
        self.log_sim_params.write(log_text)


    def init_Learner(self):
        self.init_QLearning()
        self.open_log_files()
        self.init_learning_params()
        self.init_episodes_params()
        self.init_init_pos()
        self.initTime()
        self.init_log_files()
        self.log_simulation_params()

    def log_step_time_error(self):
        text = '\r\n ERROR! Step time is too big: %.2f s' % self.step_time
        print(text)
        self.log_sim_info.write(text+'\r\n')

    def end_learning(self):
        self.get_sim_time()
        self.log_finishing_info()
        self.log_sim_data()
        self.shutdown_sim()

    def get_sim_time(self):     
         # simulation time
        self.sim_time = (rospy.Time.now() - self.t_sim_start).to_sec()
        self.sim_time_h = self.sim_time // 3600
        self.sim_time_m = ( self.sim_time - self.sim_time_h * 3600 ) // 60
        self.sim_time_s = self.sim_time - self.sim_time_h * 3600 - self.sim_time_m * 60


 
        self.stop_dt = dt.now()
        self.dt_string_stop = self.stop_dt.strftime("%d/%m/%Y %H:%M:%S")
        self.real_time_delta = (self.stop_dt - self.start_dt).total_seconds()
        self.real_time_h = self.real_time_delta // 3600
        self.real_time_m = ( self.real_time_delta - self.real_time_h * 3600 ) // 60
        self.real_time_s = self.real_time_delta - self.real_time_h * 3600 - self.real_time_m * 60


    def log_finishing_info(self):
        timestamp = self.dt_string_stop
        text_lines = [
            "---------------------------------------",
            f"\nMAX EPISODES REACHED({self.MAX_EPISODES}), LEARNING FINISHED ==> {timestamp}",
            f"\nSimulation time: {int(self.sim_time_h):02d}:{int(self.sim_time_m):02d}:{int(self.sim_time_s):02d}  h/m/s"
            f"\nReal time: {int(self.real_time_h):02d}:{int(self.real_time_m):02d}:{int(self.real_time_s):02d}  h/m/s"
        ]
        log_text = '\n'.join(text_lines) + '\n'
        print(log_text)
        self.log_sim_info.write('\n' + log_text + '\n')
        self.log_sim_params.write(log_text + '\n')


    def log_sim_data(self):

        # Log data to file
        self.saveQTable(self.LOG_FILE_DIR+'/Qtable.csv', self.Q_table)
        np.savetxt(self.LOG_FILE_DIR+'/StateSpace.csv', self.state_space, '%d')
        np.savetxt(self.LOG_FILE_DIR+'/steps_per_episode.csv', self.steps_per_episode, delimiter = ' , ')
        np.savetxt(self.LOG_FILE_DIR+'/reward_per_episode.csv', self.reward_per_episode, delimiter = ' , ')
        np.savetxt(self.LOG_FILE_DIR+'/T_per_episode.csv', self.T_per_episode, delimiter = ' , ')
        np.savetxt(self.LOG_FILE_DIR+'/EPSILON_per_episode.csv', self.EPSILON_per_episode, delimiter = ' , ')
        np.savetxt(self.LOG_FILE_DIR+'/reward_min_per_episode.csv', self.reward_min_per_episode, delimiter = ' , ')
        np.savetxt(self.LOG_FILE_DIR+'/reward_max_per_episode.csv', self.reward_max_per_episode, delimiter = ' , ')
        np.savetxt(self.LOG_FILE_DIR+'/reward_avg_per_episode.csv', self.reward_avg_per_episode, delimiter = ' , ')
        np.savetxt(self.LOG_FILE_DIR+'/t_per_episode.csv', self.t_per_episode, delimiter = ' , ')

        data = {
            'Steps per Episode': self.steps_per_episode,
            'Reward per Episode': self.reward_per_episode,
            't per Episode': self.t_per_episode,
            'T per Episode': self.T_per_episode,
            'Epsilon per Episode': self.EPSILON_per_episode,
            'Reward Min per Episode': self.reward_min_per_episode,
            'Reward Max per Episode': self.reward_max_per_episode,
            'Reward Avg per Episode': self.reward_avg_per_episode,
        }

        df = pd.DataFrame(data)
        df.to_csv(self.LOG_FILE_DIR+'/combined_data.csv', index=False)

    def shutdown_sim(self):

        # Close files and shut down node
        self.log_sim_info.close()
        self.log_sim_params.close()
        # rospy.signal_shutdown('End of learning')

    def getCrashPos(self):
        # get crash position
        odomMsg = rospy.wait_for_message('/odom', Odometry)
        ( self.x_crash , self.y_crash ) = self.getPosition(odomMsg)
        self.theta_crash = degrees(self.getRotation(odomMsg))



    def end_current_episode(self):

        self.t_ep = rospy.Time.now()
        self.reward_min = np.min(self.ep_reward_arr)
        self.reward_max = np.max(self.ep_reward_arr)
        self.reward_avg = np.mean(self.ep_reward_arr)
        self.log_episode_info()
        self.append_episode_data()


    def log_episode_info(self):
        timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"---------------------------------------\n[{timestamp}] Episode {self.episode}:"

        if self.crash:
            event_info = f"CRASH at {{X: {self.x_crash:.2f}, Y: {self.y_crash:.2f}, Theta: {self.theta_crash:.2f}}}"
        elif self.ep_steps >= self.MAX_STEPS_PER_EPISODE:
            event_info = f"MAX STEPS PER EPISODE REACHED ({self.MAX_STEPS_PER_EPISODE})"
        else:
            event_info = "UNKNOWN TERMINAL CASE"

        ep_stats = (
            f"Episode Time: {self.ep_time:.2f} s (Avg Step: {self.ep_time / self.ep_steps:.2f} s)\n"
            f"Episode Steps: {self.ep_steps}\n"
            f"Episode Reward: {self.ep_reward:.2f}\n"
            f"Episode Avg Reward : {self.reward_avg:.2f}\n"
            f"Episode Min Reward : {self.reward_min:.2f}\n"

        )

        exploration_info = f"T = {self.T:.2f}" if self.EXPLORATION_FUNCTION == 1 else f"EPSILON = {self.EPSILON:.2f}"

        log_entry = f"{header}\n{event_info}\n{ep_stats}{exploration_info}\n"
        print(log_entry)
        self.log_sim_info.write('\n' + log_entry)

    def append_episode_data(self):


        self.steps_per_episode = np.append(self.steps_per_episode, self.ep_steps)
        self.reward_per_episode = np.append(self.reward_per_episode, self.ep_reward)
        self.T_per_episode = np.append(self.T_per_episode, self.T)
        self.EPSILON_per_episode = np.append(self.EPSILON_per_episode, self.EPSILON)
        self.t_per_episode = np.append(self.t_per_episode, self.ep_time)
        self.reward_min_per_episode = np.append(self.reward_min_per_episode, self.reward_min)
        self.reward_max_per_episode = np.append(self.reward_max_per_episode, self.reward_max)
        self.reward_avg_per_episode = np.append(self.reward_avg_per_episode, self.reward_avg)
        self.ep_reward_arr = np.array([])

    def init_new_episode(self):
        
        self.ep_steps = 0
        self.ep_reward = 0
        self.crash = 0
        self.robot_in_pos = False
        self.first_action_taken = False
        if self.T > self.T_MIN:
            self.T = self.T_GRAD * self.T
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON = self.EPSILON_GRAD * self.EPSILON
        self.episode = self.episode + 1


    def set_init_pos(self):    
        # init pos
        if self.RANDOM_INIT_POS:
            ( self.x_init , self.y_init , self.theta_init ) = self.robotSetRandomPos(self.setPosPub)
        else:
            ( self.x_init , self.y_init , self.theta_init ) = self.robotSetPos(self.setPosPub, self.X_INIT, self.Y_INIT, self.THETA_INIT)
        self.odomMsg = rospy.wait_for_message('/odom', Odometry)
        ( self.x , self.y ) = self.getPosition(self.odomMsg)
        self.theta = degrees(self.getRotation(self.odomMsg))

    def check_init_pos(self):  
        # check init pos
        if abs(self.x-self.x_init) < 0.01 and abs(self.y-self.y_init) < 0.01 and abs(self.theta-self.theta_init) < 1:
            self.robot_in_pos = True
            #sleep(2)
        else:
            self.robot_in_pos = False

    def robot_get_init_pos(self):

        self.robotStop(self.velPub)
        self.ep_steps = self.ep_steps - 1
        self.first_action_taken = False
        self.set_init_pos()
        self.check_init_pos()

    


    def log_status(self):

        if self.first_action_taken:
   
            if not self.status_uqt == 'updateQTable => OK':
                print('\r\n', self.status_uqt, '\r\n')
                self.log_sim_info.write('\r\n'+self.status_uqt+'\r\n')

        # if not (self.status_strat == 'softMaxSelection => OK' or self.status_strat == 'epsiloGreedyExploration => OK'):
        #     # print('\r\n', self.status_strat, '\r\n')
        #     # self.log_sim_info.write('\r\n'+self.status_strat+'\r\n')            

        if not self.status_rda == 'robotDoAction => OK':
            print('\r\n', self.status_rda, '\r\n')
            self.log_sim_info.write('\r\n'+self.status_rda+'\r\n')


    def scan_env(self):
        ( self.lidar, self.angles ) = self.lidarScan(self.msgScan)
        ( self.state_ind, self.x1, self.x2 ,self.x3 ,self.x4 ) = self.scanDiscretization(self.state_space, self.lidar)
        self.crash = self.checkCrash(self.lidar)

    def get_action(self):
        if self.EXPLORATION_FUNCTION == 1 :
            ( self.action, self.status_strat ) =self.softMaxSelection(self.Q_table, self.state_ind, self.actions, self.T)
        else:
            ( self.action, self.status_strat ) = self.epsiloGreedyExploration(self.Q_table, self.state_ind, self.actions, self.EPSILON)

    def update_prev_state(self):
        self.prev_lidar = self.lidar
        self.prev_action = self.action
        self.prev_state_ind = self.state_ind

    def get_reward_update_qtable(self):
        ( self.reward, self.terminal_state ) = self.getReward(self.action, self.prev_action, self.lidar, self.prev_lidar, self.crash)
        ( self.Q_table, self.status_uqt ) = self.updateQTable(self.Q_table, self.prev_state_ind, self.action, self.reward, self.state_ind, self.alpha, self.gamma)
        # print(f'Q-table UPDATED at {self.t_step}')
        self.ep_reward = self.ep_reward + self.reward
        self.ep_reward_arr = np.append(self.ep_reward_arr, self.reward)


    def process_first_action_step(self):
        self.scan_env()
        self.get_action()
        self.status_rda = self.robotDoAction(self.velPub, self.action)
        self.update_prev_state()
        self.log_status()
        self.first_action_taken = True

    def process_action_step(self):

        self.scan_env() 
        self.get_reward_update_qtable()
        self.get_action()
        self.status_rda = self.robotDoAction(self.velPub, self.action)
        # self.update_prev_state()
        self.log_status()   
        self.update_prev_state()

  


    def run_learner(self):


        self.setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        self.init_Learner()

        # main loop
        while not rospy.is_shutdown():
            self.msgScan = rospy.wait_for_message('/scan', LaserScan)
            
            # Secure the minimum time interval between 2 actions
            self.step_time = (rospy.Time.now() - self.t_step).to_sec()
            if self.step_time > self.MIN_TIME_BETWEEN_ACTIONS:
                self.t_step = rospy.Time.now()
                if self.step_time > 2:
                    self.log_step_time_error()
                # End of Learning
                if self.episode > self.MAX_EPISODES:
                    self.end_learning()
                    break
                else:
                    self.ep_time = (rospy.Time.now() - self.t_ep).to_sec()
                    # End of en Episode
                    if self.crash or self.ep_steps >= self.MAX_STEPS_PER_EPISODE: 
                        self.robotStop(self.velPub)
                        if self.crash:
                            self.getCrashPos()
                        
                        self.end_current_episode()
                        self.init_new_episode() 

                    else: #Continue of the ep steps
                        self.ep_steps = self.ep_steps + 1
                        #Initial position
                        if not self.robot_in_pos:
                            self.robot_get_init_pos()
                        elif not self.first_action_taken:
                            self.process_first_action_step()
                        else:
                            self.process_action_step()



# if __name__ == '__main__':
#     try:
#         ML=MapLearner()
#         ML.run_learner()
#     except rospy.ROSInterruptException:
#         print('Simulation interrupted by the user')
#         pass


                            
             


                            














