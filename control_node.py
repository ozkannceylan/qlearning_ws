
import rospy
from tools.motion_planner import MotionPlanner    
from tools.controller import Control   
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from time import sleep
import sys
from math import *
import pandas as pd

import yaml

params_yaml_dir="/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml"
with open(params_yaml_dir, 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

DATA_PATH = params['DATA_PATH']
MODULES_PATH = params['MODULES_PATH']
sys.path.insert(0, MODULES_PATH)

min_step_time = params['control_node_params']['min_step_time']

init_x=params['control_node_params']['init_positions']['x']
init_y=params['control_node_params']['init_positions']['y']
init_theta=params['control_node_params']['init_positions']['theta']

goal_x=params['control_node_params']['goal_positions']['x']
goal_y=params['control_node_params']['goal_positions']['y']
goal_theta=params['control_node_params']['goal_positions']['theta']

path_mode=params['control_node_params']['path_mode_selection']

X_INIT = init_x[path_mode]
Y_INIT = init_y[path_mode]
THETA_INIT = init_theta[path_mode]

X_GOAL = goal_x[path_mode]
Y_GOAL = goal_y[path_mode]
THETA_GOAL = goal_theta[path_mode]


C=Control()
rospy.init_node('control_node', anonymous = False)
rate = rospy.Rate(10)
setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
experiment_name= 'softmax'
def update_text(text,x,y,theta,X_GOAL,Y_GOAL,THETA_GOAL):
    text = text + '\r\nx :       %.2f -> %.2f [m]' % (x, X_GOAL)
    text = text + '\r\ny :       %.2f -> %.2f [m]' % (y, Y_GOAL)
    text = text + '\r\ntheta :   %.2f -> %.2f [degrees]' % (degrees(theta), THETA_GOAL)
    return text

def start_test(experiment_path="/home/ozkan/Desktop/rl_ws/src/scripts/Data/Data_Logs1/"):
        # C=Control()
        # rospy.init_node('control_node', anonymous = False)
        # rate = rospy.Rate(10)
        # setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        # velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        q_table_dir = f"{experiment_path}/Qtable.csv"
        actions=C.createActions()
        state_space=C.createStateSpace()
        Q_table=C.readQTable(q_table_dir)
        # result =pd.DataFrame()
        result={}


        t_0 = rospy.Time.now()
        t_start = rospy.Time.now()

        # init timer
        while not (t_start > t_0):
            t_start = rospy.Time.now()

        t_step = t_start
        count = 0
        obstacle=0

        # robot in initial position
        robot_in_pos = False

        # because of the video recording
        sleep(1)

        # main loop
        while not rospy.is_shutdown():
            msgScan = rospy.wait_for_message('/scan', LaserScan)
            odomMsg = rospy.wait_for_message('/odom', Odometry)

            # Secure the minimum time interval between 2 actions
            step_time = (rospy.Time.now() - t_step).to_sec()

            # if step_time > min_step_time:
            #     t_step = rospy.Time.now()


            if not robot_in_pos: # robot not in initial position
                C.robotStop(velPub)
                # control.init_robot_pos()
                # Position Initializing
                ( x_init , y_init , theta_init ) = C.robotSetPos(setPosPub, X_INIT, Y_INIT, THETA_INIT)
                # check init pos
                odomMsg = rospy.wait_for_message('/odom', Odometry)
                ( x , y ) = C.getPosition(odomMsg)
                theta = degrees(C.getRotation(odomMsg))
                print(theta, theta_init)
                if abs(x-x_init) < 0.05 and abs(y-y_init) < 0.05 and abs(theta-theta_init) < 2: #Checking the robot is in the init pos
                    robot_in_pos = True
                    # MP.print_init_pos(x,y,theta)
                else:
                    robot_in_pos = False
            else: #Now we are sure the robot at the init pos
                count = count + 1
                text = '\r\nStep %d , Step time %.2f s' % (count, step_time)

                # Get real-time robot position and orientation
                ( x , y ) = C.getPosition(odomMsg)
                theta = C.getRotation(odomMsg)

                # Get real-time lidar scan data and discretize them
                ( lidar, angles ) = C.lidarScan(msgScan)
                ( state_ind, x1, x2 ,x3 ,x4 ) = C.scanDiscretization(state_space, lidar)

                # Check for objects nearby
                crash = C.checkCrash(lidar)
                object_nearby = C.checkObjectNearby(lidar)
                goal_near = C.checkGoalNear(x, y, X_GOAL, Y_GOAL)
                enable_feedback_control = True

                # Stop the simulation
                if crash:
                    C.robotStop(velPub)
                    # rospy.signal_shutdown('End of testing!')
                    text = text + ' ==> Crash! End of simulation!'
                    status = 'Crash! End of simulation!'
                    obstacle=obstacle+60
                    result["obstacle"]=obstacle
                    result["success"] = False 
                    result["experiment_duration"] = 60

                    return result
                # Feedback control algorithm
                elif enable_feedback_control and ( not object_nearby or goal_near ):
                    status = C.robotFeedbackControl(velPub, x, y, theta, X_GOAL, Y_GOAL, radians(THETA_GOAL)) 
                    text = text + ' ==> Feedback control algorithm '
                    if goal_near:
                        text = text + '(goal near)'
                # Q-learning algorithm
                else:
                    ( action, status ) = C.getBestAction(Q_table, state_ind, actions)
                    if not status == 'getBestAction => OK':
                        print('\r\n', status, '\r\n')
                    print(f"Q learning Action: {action}")
                    status = C.robotDoAction(velPub, action)
                    if object_nearby:
                        obstacle=obstacle+1
                        result["obstacle"]=obstacle
                    if not status == 'robotDoAction => OK':
                        print('\r\n', status, '\r\n')
                    text = text + ' ==> Q-learning algorithm'

                text = update_text(text,x,y,theta,X_GOAL,Y_GOAL,THETA_GOAL)
        
                if status == 'Goal position reached!':
                    C.robotStop(velPub)
                    t_end = rospy.Time.now()
                    result["experiment_duration"] = (t_end - t_start).to_sec()
                    result["success"] = True
                    # rospy.signal_shutdown('End of testing!')
                    text = text + '\r\n\r\nGoal position reached! End of simulation!'
                    return result

                print(text)
                # print(result)
        return result


if __name__ == '__main__':
    # MP=MotionPlanner()
    # C=Control()
    try:

        experiment_log_path = f"/home/ozkan/Desktop/rl_ws/src/scripts/Data/{experiment_name}/"
        for i in range(15):

            experiment_dir = f"{experiment_log_path}/Hyperparam_opt_logs_{i+1}"
            print(experiment_dir)

            result=start_test(experiment_dir)
            print(result)
            # rospy.signal_shutdown('End of hyperparameter search')

        # result=start_test()
        # print(result)

    except rospy.ROSInterruptException:
        C.robotStop(velPub)
        print('Simulation terminated!')
        pass
