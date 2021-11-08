# Author: Akifumi Wachi, Yunyue Wei
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import sys
import json

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from myenv.safety_gym_discrete import Engine_Discrete, \
    coord_to_safety_gym_pos, safety_gym_pos_to_coord

"""
A script to generate discretized safety-gym environment
"""

world_shape = (5, 5)  # size of the safety-gym environment
MAP_SIZE = 7 * world_shape[0] / 20


def check_path(map, from_pos, end_pos) -> bool:
    """Check if there is feasible path from start to the goal"""
    print('from direction(%d,%d)-->(%d,%d):'
          % (from_pos[0], from_pos[1], end_pos[0], end_pos[1]))
    row = map.shape[0]
    col = map.shape[1]
    step = 0
    direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    moveList = []
    newCanMoveList = []
    newCanMoveList.append((from_pos[0], from_pos[1]))
    while from_pos[0] != end_pos[0] \
            and from_pos[1] != end_pos[1] \
            and len(newCanMoveList) > 0:
        prevMoveList = newCanMoveList
        newCanMoveList = []
        step += 1
        for xyItem in prevMoveList:
            if xyItem not in moveList:
                moveList.append(xyItem)
            for dItem in direction:
                chgx = xyItem[0] + dItem[0]
                chgy = xyItem[1] + dItem[1]

                if chgx < 0 or chgx >= row or chgy < 0 or chgy >= col:
                    continue
                elif map[int(chgx), int(chgy)] == 1 \
                        or (chgx, chgy) in moveList:
                    continue
                elif chgx == end_pos[0] and chgy == end_pos[1]:
                    from_pos[0] = chgx
                    from_pos[1] = chgy
                    break
                else:
                    newCanMoveList.append((chgx, chgy))
                    moveList.append((chgx, chgy))

            if from_pos[0] == end_pos[0] and from_pos[1] == end_pos[1]:
                break

    if from_pos[0] != end_pos[0] and from_pos[1] != end_pos[1]:
        print("it doesn't have the way from start to end.")
        return False
    else:
        print('it need step %d from start to end.' % step)
        return True


def sample_random_region(config, coord, world_shape, step_size,
                         test=False, plot=False) -> dict:
    hazards_num = config['hazards_num']
    feasible = False
    np.random.seed()  # different under multiprocess
    while not feasible:

        # one robot, one goal and hazards_num hazards
        selected_pos = []
        selected_index = []

        map = np.zeros((world_shape[0], world_shape[1]))

        sele_coord = []

        for i in range(hazards_num + 2):
            while (True):
                rand_index = np.random.randint(0, len(coord))
                pos = coord_to_safety_gym_pos(
                    coord[rand_index],
                    world_shape,
                    step_size
                )
                sele_coord.append(coord[rand_index])
                if config['allow_conflict'] \
                        or rand_index not in selected_index:
                    selected_index.append(rand_index)
                    selected_pos.append(pos)
                    break
        print(selected_pos)

        for c in sele_coord[2:]:
            map[int(c[0]), int(c[1])] = 1

        feasible = check_path(map, sele_coord[1], sele_coord[0])

    if test == False:
        config['robot_locations'] = [list(selected_pos[1])]

    config['goal_locations'] = [list(selected_pos[0])]
    config['goal_size'] = MAP_SIZE / world_shape[0] / 2
    config['hazards_size'] = MAP_SIZE / world_shape[0] / 2
    config['hazards_locations'] = [list(x) for x in selected_pos[2:]]

    # Plot environment
    if plot:
        plt.figure()
        plt.scatter(
            [x[0] for x in config['robot_locations']],
            [x[1] for x in config['robot_locations']],
            color='red',
            s=100 / world_shape[0]
        )
        plt.scatter(
            [x[0] for x in config['goal_locations']],
            [x[1] for x in config['goal_locations']],
            color='green',
            s=100 / world_shape[0]
        )
        plt.scatter(
            [x[0] for x in config['hazards_locations']],
            [x[1] for x in config['hazards_locations']],
            color='blue',
            s=100 / world_shape[0]
        )
        plt.show()

    return config


def generate_test_env(
        NUM_LIDAR,
        view_size,
        test_world_shape,
        step_size,
        agent_pos
) -> Engine_Discrete:
    tn, tm = test_world_shape
    # reward initialization
    # initialization
    hazard_pos = [0, 1]
    actual_step = []
    actual_cost = []
    test_config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'observe_hazards': True,
        'lidar_max_dist': 1.5 * MAP_SIZE,
        'constrain_hazards': True,
        'lidar_num_bins': NUM_LIDAR,
        'hazards_num': 1,
        'placements_extents': [-MAP_SIZE / 2, -MAP_SIZE / 2,
                               MAP_SIZE / 2, MAP_SIZE / 2],
        'num_steps': 1e9,
        'observation_flatten': False,
        'render_labels': True,
        'hazards_locations': [coord_to_safety_gym_pos(hazard_pos,
                                                      world_shape, step_size)],
        'robot_locations': [agent_pos],
        'robot_rot': np.pi,
        'goal_size': MAP_SIZE / world_shape[0] / 2,
        'hazards_size': MAP_SIZE / world_shape[0] / 2,
        'goal_locations': [coord_to_safety_gym_pos([0, 0],
                                                   world_shape, step_size)],
        'allow_conflict': True,
        'continue_goal': True,
        'lidar_alias': False,
        'goal_keepout': 0.0,
        'hazards_keepout': 0.0,
        'robot_keepout': 0.0
    }

    test_env = Engine_Discrete(
        test_config,
        view_size=view_size,
        world_shape=test_world_shape,
        step_size=step_size
    )

    return test_env


def save_env(id, view_size, start_path='./') -> Engine_Discrete:
    """
    Generate environment and save as npz file
    """
    try:
        np.load(start_path + 'params/' + str(world_shape[0])
                + '/env_map' + str(id) + '.npz')
        print('env existed')
        return
    except Exception as e:
        print(id, 'no env,start generating', e)
    print('new', id)
    NUM_LIDAR = 16
    step_size = (1, 1)
    hazards_ratio = 0.1
    NUM_HAZARDS = int(world_shape[0] * world_shape[1] * hazards_ratio)
    n, m = world_shape
    step1, step2 = step_size
    xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                         np.linspace(0, (m - 1) * step2, m),
                         indexing="ij")
    coord = np.vstack((xx.flatten(), yy.flatten())).T

    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'observe_hazards': True,
        'lidar_max_dist': 1.5 * MAP_SIZE,
        'constrain_hazards': True,
        'lidar_num_bins': NUM_LIDAR,
        'hazards_num': NUM_HAZARDS,
        'placements_extents': [-MAP_SIZE / 2, -MAP_SIZE / 2,
                               MAP_SIZE / 2, MAP_SIZE / 2],
        'num_steps': 1e9,
        'observation_flatten': False,
        'render_labels': True,
        'robot_rot': 0,
        'goal_size': MAP_SIZE / world_shape[0] / 2,
        'hazards_size': MAP_SIZE / world_shape[0] / 2,
        'allow_conflict': False,
        'continue_goal': False,
        'lidar_alias': False,
        'goal_keepout': 0.0,
        'hazards_keepout': 0.0,
        'robot_keepout': 0.0
    }
    config = sample_random_region(config, coord, world_shape,
                                  step_size, plot=False)
    env = Engine_Discrete(
        config,
        world_shape,
        view_size=view_size,
        step_size=step_size,
        link=lambda x: 1 / (1 + np.exp(-x))
    )
    env.get_info()

    with open(start_path + 'params/' + str(world_shape[0]) + '/env_settings'
              + str(id) + '.json', 'w',encoding='utf8') as fp:
        json.dump(config, fp, indent=True)
    np.savez(start_path + 'params/' + str(world_shape[0])
             + '/env_map' + str(id) + '.npz',
             feature=env.feature, safety=env.safety, reward_fun=env.reward_fun)

    return env


if __name__ == '__main__':
    for i in range(11):
        save_env(i, view_size=5)
