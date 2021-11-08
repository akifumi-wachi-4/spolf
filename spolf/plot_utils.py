# Author: Akifumi Wachi
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

import copy
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt


class SPOLF_Plot(object):
    def __init__(self) -> None:
        self.dif_theta_reward = []
        self.dif_theta_safety = []
        self.lambda_max = []
        self.reward_history = []
        self.cost_history = []
        self.plot_reward_history = []
        self.unsafe_cnt = 0

    def update(
        self, env, x, initialization=False, chosen_state=None
        ) -> None:
        if 'gp' in x.sim_type:
            None
        else:
            _norm_dif_theta_reward = np.linalg.norm(
                env.theta_reward - x.theta_hat_reward, 2
                )
            self.dif_theta_reward.append(_norm_dif_theta_reward)

            _norm_dif_theta_safety = np.linalg.norm(
                env.theta_safety - x.theta_hat_safety, 2
                )
            self.dif_theta_safety.append(_norm_dif_theta_safety)

            self.lambda_max.append(sqrt(x.eig_max_inv_W))

        if initialization:
            self.reward_history.append(
                env.reward[chosen_state[0], chosen_state[1]]
                )
        else:
            self.reward_history.append(
                env.reward[env.agent_pos[0], env.agent_pos[1]]
                )

        if env.safety[env.agent_pos[0], env.agent_pos[1]] > env.h:
            None
        else:
            self.unsafe_cnt += 1

    def update_safety_gym(
        self, env, x, reward, initialization=False
        ) -> None:
        if 'gp' in x.sim_type:
            None
        else:
            self.lambda_max.append(sqrt(x.eig_max_inv_W))

        act = [0., 0.]
        cost = (1 - env.cost()['cost'])

        self.reward_history.append(reward)
        self.cost_history.append(cost)
        if initialization == False:
            if cost <= env.h:
                self.unsafe_cnt += 1

    def print_color(self, text, color) -> None:
        color_dict = {
            'RED': '\033[31m',
            'GREEN': '\033[32m',
            'YELLOW': '\033[33m',
            'BLUE': '\033[34m',
            'MAGENTA': '\033[35m',
            'CYAN': '\033[36m'
            }
        print(color_dict[color] + text + '\033[0m')

    def get_plot_reward_history(self):
        num_history = 10
        _reward = copy.copy(self.reward_history)
        for i in range(1, len(self.reward_history)):
            self.plot_reward_history.append(
                sum(_reward[max(0, i-num_history):i]) / min(i, num_history)
                )

    def plot_result(
        self,
        plot_theta=True,
        plot_lambda_max=True,
        plot_reward=True
        ) -> None:
        ''' Plot the simulation results '''
        if plot_theta:
            plt.figure()
            plt.title('Predicted coefficients for reward and safety')
            plt.plot(self.dif_theta_reward, 'r-')
            plt.plot(self.dif_theta_safety, 'b--')
            plt.yscale('log')
            plt.show()

        if plot_lambda_max:
            plt.figure()
            plt.title('Maximum eigen value of $W^{-1}$')
            plt.plot(self.lambda_max)
            plt.ylim([0, 1.05])
            plt.show()

        if plot_reward:
            plt.figure()
            plt.title('Averaged Reward')
            plt.plot(self.plot_reward_history)
            plt.ylim([0, 1.05])
            plt.show()

    def show_map(self, target_map, fig_title, contour_type='binary') -> None:
        plt.figure()
        plt.title(fig_title)
        if contour_type == 'binary':
            plt.imshow(target_map.T, interpolation='none', cmap='gray')
        else:
            plt.contourf(target_map.T)
            plt.colorbar()
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.xaxis.tick_top()
            ax.yaxis.tick_left()
        plt.show()

    def print_result(self, env, i, print_interval=1) -> None:
        if (i+1) % print_interval == 0:
            _reward = round(env.reward[env.agent_pos[0], env.agent_pos[1]], 3)
            _message_pos = str(i) + ', ' + str(env.agent_pos)
            _message_reward = ', ' + 'reward = ' + str(_reward)
            _message_unsafe = ', ' + 'unsafe count = ' + str(self.unsafe_cnt)
            _message = _message_pos + _message_reward + _message_unsafe
            if env.safety[env.agent_pos[0], env.agent_pos[1]] > env.h:
                self.print_color(_message, 'GREEN')
            else:
                self.print_color(_message, 'RED')

    def print_result_safety_gym(
        self, env, i, actual_cost, reward, env_id, print_interval=1
        ) -> None:
        if (i + 1) % print_interval == 0:
            _message_pos = ' step ' + str(i) + ', ' + str(env.agent_pos)
            _message_reward = ', ' + 'reward = ' + str(reward)
            _message_unsafe = ', ' + 'unsafe count = ' + str(self.unsafe_cnt)
            _message_cost = ', ' + 'cost = ' + str(np.sum(actual_cost))

            _message = str(env_id) + _message_pos + _message_reward
            _message += _message_unsafe + _message_cost
            if env.safety[env.agent_pos[0], env.agent_pos[1]] > env.h:
                self.print_color(_message, 'GREEN')
            else:
                self.print_color(_message, 'RED')


def redraw(window, env, img, x, cfg) -> None:
    if not cfg.agent.agent_view:
        img = env.render('rgb_array', tile_size=cfg.env.tile_size)
    window.show_img(img, env, x)


def reset(window, env, obs, x, cfg) -> None:
    if cfg.env.seed != -1:
        env.seed(cfg.env.seed)
    obs = env.reset()
    if cfg.env.render:
        window.set_caption(env.mission)
        redraw(window, env, obs, x, cfg)
    window.close()
