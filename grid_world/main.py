# Author: Akifumi Wachi
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import sys
import gym
import hydra
import numpy as np

from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from omegaconf import DictConfig

import myenv
from myenv.window import MyWindow

sys.path.append('../spolf')
from agent import SPOLF_Agent
from plot_utils import SPOLF_Plot, redraw, reset


@hydra.main(config_path="./", config_name="config")
def main(cfg: DictConfig) -> None:

    env = gym.make(cfg.env.env_name)

    if cfg.env.reuse_env:
        env.reuse_env_params(
            file_name=hydra.utils.to_absolute_path(cfg.env.env_param_file)
            )

    # This part is not needed??
    if cfg.agent.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    x = SPOLF_Agent(env, cfg, h=env.h)
    plot_x = SPOLF_Plot()

    # Initialize the prior information on reward and safety functions
    for _ in range(cfg.agent.num_init_info):
        chosen_state = x.initialization(env, cfg)
        plot_x.update(env, x, initialization=True, chosen_state=chosen_state)

    window = MyWindow('gym_minigrid - ' + cfg.env.env_name)

    plot_x.print_color('Agent Type: ' + cfg.sim_type, 'BLUE')
    for i in range(cfg.agent.num_timestep):
        if cfg.sim_type == 'random':
            action = np.random.choice(
                [env.actions.left, env.actions.right, env.actions.forward]
                )
            obs, context, reward, safety, _, _ = env.step(action)

        else:
            next_state = x.optimize_policy(env, cfg)
            if cfg.sim_type == 'safe_glm' and cfg.agent.stack_workaround != 'None':
                # Event-triggered Safe Expansion (ETSE) algorithm
                if x.chck_contradiction(env, next_state):
                    plot_x.print_color(
                        str(i) + ', (' + cfg.agent.stack_workaround + ')',
                        color='MAGENTA'
                        )
                    for _ in range(cfg.agent.num_stack_workaround):
                        next_state = x.optimize_policy(
                            env=env,
                            cfg=cfg,
                            stack_workaround=cfg.agent.stack_workaround
                            )
                        action_list = env.get_action_from_state(next_state)
                        for _a in action_list:
                            obs, context, reward, safety, _, _ = env.step(_a)

                        x.update(env, cfg, obs, context, reward, safety)
                        plot_x.update(env, x)

            action_list = env.get_action_from_state(next_state)
            for _a in action_list:
                obs, context, reward, safety, _, _ = env.step(_a)

        x.update(env, cfg, obs, context, reward, safety)
        plot_x.update(env, x)
        plot_x.print_result(env, i)

        if cfg.env.render:
            redraw(window, env, obs, x, cfg)

    reset(window, env, obs, x, cfg)

    # Blocking event loop
    if cfg.env.render:
        window.show(block=True)

    plot_x.get_plot_reward_history()
    _default_fn = 'results/' + str(cfg.idx_sim) + '/sim_result_' + cfg.sim_type
    if cfg.sim_type == 'safe_glm' and cfg.agent.stack_workaround == 'None':
        npz_file_name = _default_fn + '_no_ETSE'
    else:
        npz_file_name = _default_fn

    np.savez(
        hydra.utils.to_absolute_path(npz_file_name),
        reward_history=plot_x.plot_reward_history,
        unsafe_cnt=plot_x.unsafe_cnt
        )

    # Plot the results
    if cfg.display.plot_result:
        plot_x.plot_result()
        plot_x.show_map(x.is_visited_map, fig_title='visited map')
        plot_x.show_map(x.is_observed_map, fig_title='observation map')
        plot_x.show_map(env.true_safety_map, fig_title='true safety map')
        plot_x.show_map(
            x.safety_reachability_map[:, :, 0],
            fig_title='optimistic safety map w/ reachability'
            )
        plot_x.show_map(
            x.safety_reachability_map[:, :, 1],
            fig_title='pessimistic safety map w/ reachability'
            )

    # Save the current environment settings
    if not cfg.env.reuse_env:
        np.savez(
            hydra.utils.to_absolute_path(
                    'myenv/params/' + str(cfg.idx_sim) + '/env_settings'
                    ),
            context=env.context,
            reward=env.reward, theta_reward=env.theta_reward,
            safety=env.safety, theta_safety=env.theta_safety,
            true_safety_map=env.true_safety_map
            )


if __name__ == "__main__":
    main()
