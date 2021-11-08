# Author: Akifumi Wachi, Yunyue Wei
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import json
import os

import hydra
from omegaconf import (DictConfig, OmegaConf)

from algos import cpo_agent, ppo_lag_agent, trpo_lag_agent
from run_exp import SafetyGymExp


def run_baseline():
    cfg = OmegaConf.load('config.yaml')
    env_id = cfg.idx_sim
    algo = cfg.sim_type
    print(env_id)
    try:
        print(os.getcwd())
        start_path = os.getcwd()[:os.getcwd().index('safety_gym_discrete')+20]
        world_shape = (cfg.env.size, cfg.env.size)
        # env_id = 1
        with open(
            start_path + 'myenv/params/' + str(world_shape[0])
            + '/env_settings' + str(env_id) + '.json', 'r',
            encoding='utf8'
            ) as fp:
            config = json.load(fp)

        config["observation_flatten"] = True
        del config['allow_conflict']
        config['reward_distance'] = cfg.env.reward_distance
        config['lidar_num_bins'] = int(cfg.env.dim_context/2)

        if algo == 'cpo':
            agent = cpo_agent()
            exp_name = 'CPO'

        elif algo == 'ppo':
            agent = ppo_lag_agent()
            exp_name = 'PPO'

        elif algo == 'trpo':
            agent = trpo_lag_agent()
            exp_name = 'TRPO'

        else:
            NotImplementedError

        exp = SafetyGymExp(
            config=config,
            world_shape=world_shape,
            agent=agent,
            ac_kwargs=dict(hidden_sizes=(64, 64),
                           ),
            steps_per_epoch=cfg.agent.num_episode,
            gamma=cfg.agent.gamma,
            cost_gamma=cfg.agent.gamma,
            logger_kwargs=dict(
                output_dir='./result/' + str(world_shape[0])
                + '/' + str(env_id) + '/' + exp_name,
                exp_name='CPO' + str(env_id),
                ),
            render=cfg.env.render
        )
        exp.main_loop(
            test_num=cfg.agent.num_init_info,
            epochs=cfg.agent.num_epoch,
            max_ep_len=cfg.agent.num_episode,
        )
    except Exception as e:
        print(e)


if __name__ == '__main__':
    run_baseline()
