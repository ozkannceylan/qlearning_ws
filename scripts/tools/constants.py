
import numpy as np

import yaml

params_yaml_dir = "/home/ozkan/Desktop/rl_ws/src/scripts/params.yaml"
with open(params_yaml_dir, 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

STATE_SPACE_IND_MAX = params['qlearning_params']['STATE_SPACE_IND_MAX']
STATE_SPACE_IND_MIN = params['qlearning_params']['STATE_SPACE_IND_MIN']
ACTIONS_IND_MAX = params['qlearning_params']['ACTIONS_IND_MAX']
ACTIONS_IND_MIN = params['qlearning_params']['ACTIONS_IND_MIN']
T_MIN = params['qlearning_params']['T_MIN']


MAX_LIDAR_DISTANCE=1.0
COLLISION_DISTANCE=0.14
NEARBY_DISTANCE=0.45
ZONE_0_LENGTH=0.4
ZONE_1_LENGTH=0.7
ANGLE_MAX=359
ANGLE_MIN=0
HORIZON_WIDTH=75

