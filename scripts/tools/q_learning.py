#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from itertools import product
from sensor_msgs.msg import LaserScan
import yaml
import os
from tools.lidar_scanner import Lidar
from tools.constants import *

class QLearning(Lidar):
    def __init__(self,params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml"):
        super().__init__(params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml")
        self.params_yaml_dir = params_yaml_dir

    def createActions(self):
        actions = np.array([0,1,2])
        return actions

    def createStateSpace(self):
        x1 = (0, 1, 2)  
        x2 = (0, 1, 2)
        x3 = (0, 1, 2, 3)
        x4 = (0, 1, 2, 3)
        # Directly create a list from the product, which ensures ordered combinations
        state_space = list(product(x1, x2, x3, x4))
        # Convert the list to a numpy array
        return np.array(state_space)


    
    def createQTable(self,n_states, n_actions):
        Q_table = np.zeros((n_states, n_actions))
        return Q_table
    
    def readQTable(self,path):
        Q_table = np.genfromtxt(path, delimiter = ' , ')
        return Q_table
    
    @staticmethod
    def saveQTable(path, Q_table):
        np.savetxt(path, Q_table, delimiter = ' , ')

    @staticmethod
    def getBestAction(Q_table, state_ind, actions):
        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            status = 'getBestAction => OK'
            a_ind = np.argmax(Q_table[state_ind,:])
            a = actions[a_ind]
        else:
            status = 'getBestAction => INVALID STATE INDEX'
            n_actions = len(actions)
            a_ind = np.random.randint(n_actions)
            a= actions[a_ind]
        return ( a, status )
      
    @staticmethod
    def getRandomAction(actions):
        n_actions = len(actions)
        a_ind = np.random.randint(n_actions)
        return actions[a_ind]
    
    def epsiloGreedyExploration(self,Q_table, state_ind, actions, epsilon):
        if np.random.uniform() > epsilon and STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            status = 'epsiloGreedyExploration => OK'
            ( a, status_gba ) = self.getBestAction(Q_table, state_ind, actions)
            if status_gba == 'getBestAction => INVALID STATE INDEX':
                status = 'epsiloGreedyExploration => INVALID STATE INDEX'
        else:
            status = 'epsiloGreedyExploration =>a OK'
            a = self.getRandomAction(actions)

        return ( a, status )
    
    def softMaxSelection(self,Q_table, state_ind, actions, T):
        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            status = 'softMaxSelection => OK'
            n_actions = len(actions)
            P = np.zeros(n_actions)

            # Boltzman distribution
            P = np.exp(Q_table[state_ind,:] / T) / np.sum(np.exp(Q_table[state_ind,:] / T))

            if T < self.T_MIN or np.any(np.isnan(P)):
                ( a, status_gba ) = self.getBestAction(Q_table, state_ind, actions)
                if status_gba == 'getBestAction => INVALID STATE INDEX':
                    status = 'softMaxSelection => INVALID STATE INDEX'
            else:
                rnd = np.random.uniform()
                status = 'softMaxSelection => OK'
                if P[0] > rnd:
                    a = 0
                elif P[0] <= rnd and (P[0] + P[1]) > rnd:
                    a = 1
                elif (P[0] + P[1]) <= rnd:
                    a = 2
                else:
                    status = 'softMaxSelection => Boltzman distribution error => getBestAction '
                    status = status + '\r\nP = (%f , %f , %f) , rnd = %f' % (P[0],P[1],P[2],rnd)
                    status = status + '\r\nQ(%d,:) = ( %f, %f, %f) ' % (state_ind,Q_table[state_ind,0],Q_table[state_ind,1],Q_table[state_ind,2])
                    ( a, status_gba ) = self.getBestAction(Q_table, state_ind, actions)
                    if status_gba == 'getBestAction => INVALID STATE INDEX':
                        status = 'softMaxSelection => INVALID STATE INDEX'
        else:
            status = 'softMaxSelection => INVALID STATE INDEX'
            a = self.getRandomAction(actions)

        return ( a, status )
    @staticmethod
    def getReward(action, prev_action, lidar, prev_lidar, crash):
        if crash:
            terminal_state = True
            reward = -100
        else:
            lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
            prev_lidar_horizon = np.concatenate((prev_lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],prev_lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
            terminal_state = False
            # Reward from action taken = fowrad -> +0.2 , turn -> -0.1
            if action == 0:
                r_action = +0.4
            else:
                r_action = -0.2
            # Reward from crash distance to obstacle change
            W = np.linspace(0.9, 1.1, len(lidar_horizon) // 2)
            W = np.append(W, np.linspace(1.1, 0.9, len(lidar_horizon) // 2))
            if np.sum( W * ( lidar_horizon - prev_lidar_horizon) ) >= 0:
                r_obstacle = +0.4
            else:
                r_obstacle = -0.4
            # Reward from turn left/right change
            if ( prev_action == 1 and action == 2 ) or ( prev_action == 2 and action == 1 ):
                r_change = -0.4
            else:
                r_change = 0.0

            # Cumulative reward
            reward = r_action + r_obstacle + r_change

        return ( reward, terminal_state )
    
    
    @staticmethod
    def updateQTable(Q_table, state_ind, action, reward, next_state_ind, alpha, gamma):
        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX and STATE_SPACE_IND_MIN <= next_state_ind <= STATE_SPACE_IND_MAX:
            status = 'updateQTable => OK'
            Q_table[state_ind, action] = (1 - alpha) * Q_table[state_ind, action] + alpha * (reward + gamma * np.max(Q_table[next_state_ind,:]))
        else:
            status = 'updateQTable => INVALID STATE INDEX'

        return ( Q_table, status )
        

