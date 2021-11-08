# Author: Akifumi Wachi, Yunyue Wei
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import json
import time
import sys
import os
from multiprocessing import Pool

import hydra
import numpy as np

from myenv.safety_gym_discrete import (
    Engine_Discrete, coord_to_safety_gym_pos, safety_gym_pos_to_coord
    )
from myenv.generate_env import save_env, generate_test_env
sys.path.append('../spolf')
from agent import SPOLF_Agent_Safety_Gym
from plot_utils import SPOLF_Plot, redraw, reset
from omegaconf import OmegaConf

# read configuration file
cfg = OmegaConf.load('config.yaml')
if cfg.env.seed != -1:
    np.random.seed(cfg.env.seed)
view_size = cfg.env.view_size
render = cfg.env.render
world_shape = (cfg.env.size, cfg.env.size)
MAP_SIZE = 7 * world_shape[0] / 20
NUM_LIDAR = int(cfg.env.dim_context/2)
step_size = (1, 1)


def check_estimation(
    bounds_safety,
    bounds_reward,
    safety,
    reward,
    h,
    world_shape,
    env_id,feature
    ) -> (bool, bool):

    est_max = [0, 0]
    true_max = [0, 0]
    hazard_pos = []
    for i in range(world_shape[0]):
        for j in range(world_shape[1]):
            safety_est = True
            reward_est = True

            if bounds_reward[est_max[0], est_max[1], 0] \
                    < bounds_reward[i, j, 0]:
                est_max = [i, j]
            if reward[i, j] == 1:
                true_max = [i, j]

            if safety[i, j] == 0:
                hazard_pos.append([i, j])

            if bounds_safety[i, j, 1] > h and safety[i, j] == 0:
                safety_est = False
                print(
                    '\033[31m', env_id, 'unsafe', i, j, bounds_safety[i, j, 0],
                    bounds_safety[i, j, 1], safety[i, j], safety_est, '\033[0m'
                    )
                print(('\033[31m', 'feat', feature[i,j], '\033[0m'))

            if bounds_reward[i, j, 0] < reward[i, j] \
                    or bounds_reward[i, j, 1] > reward[i, j]:
                reward_est = False

    print(
        'estimate', est_max,
        bounds_reward[est_max[0], est_max[1], 0],
        bounds_reward[est_max[0], est_max[1], 1]
        )
    print(
        'true', true_max,
        bounds_reward[true_max[0], true_max[1], 0],
        bounds_reward[true_max[0], true_max[1], 1]
        )

    return safety_est, reward_est


def agent_initialization(
    x, plot_x, env_id, env, cfg, view_size
    ) -> (SPOLF_Agent_Safety_Gym, SPOLF_Plot):
    # Initialize the prior information on reward and safety functions
    test_world_shape = world_shape
    hazard_pos = [0, 1]

    safety_flag = False
    reward_flag = False
    while (not safety_flag or not reward_flag):
        for test_num in range(cfg.agent.num_init_info):
            # test env to give prior information
            i = np.random.uniform(-1, 1)
            j = np.random.uniform(-1, 1)
            for k in range(2):
                if k == 0:
                    desire_pos = coord_to_safety_gym_pos(
                        [hazard_pos[0] + i, hazard_pos[1] + j],
                        test_world_shape,
                        (1, 1)
                        )
                else:
                    desire_pos = coord_to_safety_gym_pos(
                        [i, j], test_world_shape,
                        (1, 1)
                        )
                test_env = generate_test_env(
                    NUM_LIDAR,
                    view_size,
                    test_world_shape,
                    step_size,
                    desire_pos
                    )
                # test_env.render()
                act = [0, 0]
                obs, reward, done, info = test_env.step(act)
                # print('pos', i, j)
                # print('obs',np.max(obs['goal_lidar']),
                #         np.max(obs['hazards_lidar']))
                # print('reward:',reward)
                # print('cost', info['cost'])
                x.initialization_safety_gym(test_env)
                plot_x.update_safety_gym(
                    test_env, x, reward, initialization=True
                    )

        print('validation')
        # validation using fixed feature vector
        for k in range(2):
            feat = []
            if k == 0:
                for v_num in range(21):
                    feat.append([0.05 * v_num, 1])
            else:
                for v_num in range(21):
                    feat.append([1, 0.05 * v_num])

            for v_num in range(21):
                feat_1 = np.hstack((feat[v_num], [1]))
                r_result = x.link(np.dot(feat_1.T, x.theta_hat_reward))
                s_result = x.link(np.dot(feat_1.T, x.theta_hat_safety))
                if k == 0:
                    if v_num < 17:
                        if abs(r_result - 0) < 0.1:
                            reward_flag = True
                        else:
                            reward_flag = False
                            break
                    else:
                        if abs(r_result - 1) < 0.1:
                            reward_flag = True
                        else:
                            reward_flag = False
                            break
                    if abs(s_result - 0) < 0.1:
                        safety_flag = True
                    else:
                        safety_flag = False
                        break
                else:
                    if abs(r_result - 1) < 0.1 and reward_flag:
                        reward_flag = True
                    else:
                        reward_flag = False
                        break

                    if v_num  < 17:
                        if abs(s_result - 1) < 0.1 and safety_flag:
                            safety_flag = True
                        else:
                            safety_flag = False
                            break
                    else:
                        if abs(s_result - 0) < 0.1 and safety_flag:
                            safety_flag = True
                        else:
                            safety_flag = False
                            break

        if not reward_flag or not safety_flag:
            print(
                '\033[31m', env_id,
                'not successfully initialized, resampling', '\033[0m'
                )
            print(reward_flag, safety_flag)
            x = SPOLF_Agent_Safety_Gym(env, cfg, h=env.h)
        else:
            print(
                '\033[32m', env_id,
                'successfully initialized, begin explore', '\033[0m'
                )
    print('finish init')

    return x, plot_x

def load_env(start_path, world_shape, env_id, view_size) -> Engine_Discrete:
    _ws = str(world_shape[0])
    _setting_name = str(
        start_path + 'myenv/params/' + _ws + '/env_settings' + str(env_id)
        )

    _map_name = str(
        start_path + 'myenv/params/' + _ws + '/env_map' + str(env_id)
    )

    with open(_setting_name + '.json', 'r', encoding='utf8') as fp:
        config = json.load(fp)
    config['reward_distance'] = 0.01
    config['lidar_num_bins'] = 16
    config['reward_distance'] = 0.01
    env = Engine_Discrete(
        config,
        world_shape,
        step_size=step_size,
        view_size=view_size,
        link=lambda x: 1 / (1 + np.exp(-x))
        )
    info = np.load(_map_name + '.npz')
    env.feature = info['feature']
    env.safety = info['safety']
    print(env_id, np.max(env.safety[env.safety != 1]))
    env.safety[env.safety != 1] = 0
    
    env.reward_fun = info['reward_fun']
    # for old environment, the dimension of feature is only 2,
    # we need to add a constant
    if env.feature.shape[2] == 2:
        new_feature = np.random.rand(
            env.feature.shape[0], env.feature.shape[1], 3
            )
        for i in range(env.feature.shape[0]):
            for j in range(env.feature.shape[1]):
                new_feature[i, j] = np.r_[env.feature[i, j], 1]
        env.feature = new_feature

    return env


def main(env_id) -> None:
    try: # Load existing environment
        print('Load env',env_id)
        start_path = ''
        env = load_env(start_path, world_shape, env_id, view_size)
    except: # the environment does not exist, create a new one
        env = save_env(env_id, view_size, start_path='./myenv/')

    x = SPOLF_Agent_Safety_Gym(env, cfg, h=env.h)
    plot_x = SPOLF_Plot()
    actual_step = []
    actual_cost = []
    # Initialize the prior information
    # on reward and safety functions using test environment
    x, plot_x = agent_initialization(
        x, plot_x, env_id, env, cfg, view_size
        )

    # Formal environment
    plot_x.print_color('Agent Type: ' + cfg.sim_type, 'BLUE')
    for i in range(cfg.agent.num_timestep):
        # only implement safe-glm now
        if cfg.sim_type == 'safe_glm':
            start_time = time.time()
            next_state = x.optimize_policy(env, cfg)
            print('get state using', time.time() - start_time)
            if cfg.sim_type == 'safe_glm' \
                    and cfg.agent.stack_workaround != 'None':
                # Event-triggered Safe Expansion (ETSE) algorithm
                if x.chck_contradiction(env, next_state):
                    _work_around = cfg.agent.stack_workaround
                    plot_x.print_color(
                        str(i) + ', Work Around (' + _work_around + ')'
                        , 'MAGENTA'
                        )
                    for _ in range(cfg.agent.num_stack_workaround):
                        start_time = time.time()
                        next_state = x.optimize_policy(
                            env, cfg,
                            stack_workaround=cfg.agent.stack_workaround
                            )
                        obs, feature, safety, reward, total_step, total_cost \
                            = env.discreate_step([next_state[0],
                                                  next_state[1]],
                                                 render=render)
                        actual_step.append(total_step)
                        actual_cost.append(total_cost)
                        env.done = False
                        x.update(env, cfg, obs, feature, reward, safety)
                        plot_x.update_safety_gym(env, x, reward)
                        check_estimation(
                            x.bounds_safety, x.bounds_reward,
                            env.safety, env.reward_fun, env.h,
                            env.world_shape, env_id, env.feature
                            )

                        plot_x.print_result_safety_gym(env, i, actual_cost,
                                                       reward, env_id)
                        if env.goal_met():
                            print('met!')
                            break

            env.done = False
            obs, feature, safety, reward, total_step, total_cost \
                = env.discreate_step([next_state[0],
                                      next_state[1]], render=render)
            check_estimation(
                x.bounds_safety, x.bounds_reward, env.safety, env.reward_fun,
                env.h, env.world_shape, env_id, env.feature
                )
            actual_step.append(total_step)
            actual_cost.append(total_cost)

        else:
            NotImplementedError

        x.update(env, cfg, obs, feature, reward, safety)
        env.done = False
        plot_x.update_safety_gym(env, x, reward)
        plot_x.print_result_safety_gym(env, i, actual_cost,reward, env_id)
        if env.goal_met():
            print('met!')
            break

    # Save results
    save_path = 'results/' + str(world_shape[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if cfg.sim_type == 'safe_glm' and cfg.agent.stack_workaround == 'None':
        npz_file_name = save_path + '/sim_result_' + str(env_id) + '_no_ETSE'
    else:
        npz_file_name = save_path + '/sim_result_' + str(env_id)

    np.savez(
        hydra.utils.to_absolute_path(npz_file_name),
        reward_history=plot_x.plot_reward_history,
        unsafe_cnt=plot_x.unsafe_cnt,
        actual_step=actual_step,
        actual_cost=actual_cost
        )

    if cfg.display.plot_result:
        env.safety_map = env.safety > env.h
        env.true_safety_map = env._safety_reachablity(
            env.safety_map, env.agent_start_pos
            )

        plot_x.plot_result()
        plot_x.show_map(x.is_visited_map, fig_title='visited map')
        plot_x.show_map(x.is_observed_map, fig_title='observation map')
        plot_x.show_map(env.true_safety_map, fig_title='true safety map')
        plot_x.show_map(x.safety_reachability_map[:, :, 0],
                        fig_title='optimistic safety map w/ reachability')
        plot_x.show_map(x.safety_reachability_map[:, :, 1],
                        fig_title='pessimistic safety map w/ reachability')




if __name__ == '__main__':
    main(env_id=cfg.idx_sim)

