# Author: Akifumi Wachi
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import numpy as np
import copy
import hydra
import yaml
from gym_minigrid.minigrid import Grid, MiniGridEnv
from gym_minigrid.register import register


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment with neither obstacles nor goal
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        view_size=7,
        dim_feature=5,
        link=lambda x: x,
        kappa=1,
        L_mu=1,
        sigma=0.01,
        h=0.10
    ) -> None:

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.size = size

        # True safe space
        self.true_safety_map = np.zeros((self.size, self.size))

        # Map of agent direction indices to vectors
        self.DIR_TO_VEC = [
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
            # Up (negative Y)
            np.array((0, -1)),
            ]

        # Link function
        self.link = link
        self.kappa = kappa
        self.L_mu = L_mu

        # feature
        self.dim_feature = dim_feature
        self.feature = self._get_feature(size)

        # Generate reward and safety functions
        self.sigma = sigma
        self.h = h
        self.safety, self.theta_safety = self._get_reward_safety_functions(
                size=size, func_type='safety'
                )
        self.reward, self.theta_reward = self._get_reward_safety_functions(
                size=size, func_type='reward'
                )

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            see_through_walls=True,
            agent_view_size=view_size
        )

    def _get_feature(self, size) -> np.ndarray:
        """
        Get the feature vector.
        Note that the L2 norm of the feature is <= 1.
        """
        _feature = np.random.rand(size, size, self.dim_feature)
        _scale_factor = np.random.rand(size, size)
        for i in range(size):
            for j in range(size):
                _feature[i, j] /= np.linalg.norm(_feature[i, j], ord=2)
                _feature[i, j] *= _scale_factor[i, j]
        return _feature

    def _safety_reachablity(self, safety_map, initial_nodes) -> np.ndarray:
        checked = np.zeros((self.size, self.size), dtype=bool)
        checked[initial_nodes[0], initial_nodes[1]] = True

        stack = []
        stack.append(initial_nodes)

        while stack:
            node = stack.pop(0)
            adjacent = node + np.array(self.DIR_TO_VEC)
            for adj in adjacent:
                if not checked[adj[0], adj[1]] and safety_map[adj[0], adj[1]]:
                    checked[adj[0], adj[1]] = True
                    stack.append(adj)

        return checked

    def is_wall(self, i, j) -> bool:
        if i == 0 or i == self.size-1 or j == 0 or j == self.size-1:
            return True
        else:
            return False

    def _get_reward_safety_functions(self, size, func_type) -> np.ndarray:
        while True:
            # Coefficient for reward/safety functions
            _theta = np.random.rand(self.dim_feature)
            _theta /= np.linalg.norm(_theta, ord=2)
            # reward/safety funciton values
            _func_value = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    _func_value[i, j] = self.link(
                            np.inner(self.feature[i, j], _theta)
                            )

            if func_type == 'reward':
                break
            elif func_type == 'safety':
                _safety_map = _func_value > self.h
                for i in range(self.size):
                    for j in range(self.size):
                        if self.is_wall(i, j):
                            _safety_map[i, j] = False

                _init_safety = _safety_map[
                        self.agent_start_pos[0], self.agent_start_pos[1]
                        ]
                # If the starting position is not safe or the size of safe,
                # reachable space is too small, then create a new enviornment
                self.true_safety_map = self._safety_reachablity(
                    _safety_map, self.agent_start_pos
                    )

                if (np.sum(self.true_safety_map) > 100) and _init_safety:
                    break
                else:
                    self.feature = self._get_feature(size)

        return _func_value, _theta

    def _gen_grid(self, width, height) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Mission description
        self.mission = "SPOLF"

    def get_action_from_state(self, next_state) -> list:
        diff_state = [next_state[0] - self.agent_pos[0],
                      next_state[1] - self.agent_pos[1]]

        _action = []
        if np.linalg.norm(diff_state, 2) > 1:
            raise NameError('Distance between state and next state is > 1')
        elif np.linalg.norm(diff_state, 2) == 0:
            _action.append(self.actions.right)
        else:
            appropriate_dir = 0
            for i in range(4):
                if np.linalg.norm(diff_state - self.DIR_TO_VEC[i], 2) < 0.01:
                    appropriate_dir = i
            _agent_dir = copy.copy(self.agent_dir)
            while(True):
                if _agent_dir - appropriate_dir > 0:
                    _action.append(self.actions.left)
                    _agent_dir -= 1
                elif _agent_dir - appropriate_dir < 0:
                    _action.append(self.actions.right)
                    _agent_dir += 1
                else:
                    break
            _action.append(self.actions.forward)

        return _action

    def step(self, action):
        self.step_count += 1

        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

        else:
            assert False, "unknown action"

        # feature
        feature = {'observed_states': [], 'feature': []}
        topX, topY, botX, botY = self.get_view_exts()
        for i_x in range(topX, botX):
            for i_y in range(topY, botY):
                if i_x >= 1 and i_x <= self.size-2:
                    if i_y >= 1 and i_y <= self.size-2:
                        feature['observed_states'].append([i_x, i_y])
                        feature['feature'].append(self.feature[i_x, i_y])

        # reward and safety functions
        _pos = self.agent_pos
        noise_reward = np.random.normal(loc=0, scale=self.sigma)
        reward = self.reward[_pos[0], _pos[1]] + noise_reward
        noise_safety = np.random.normal(loc=0, scale=self.sigma)
        safety = self.safety[_pos[0], _pos[1]] + noise_safety

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, feature, reward, safety, done, {}

    def reuse_env_params(self, file_name) -> None:
        npz_file = np.load(file_name)
        self.feature = npz_file['feature']
        self.safety = npz_file['safety']
        self.theta_safety = npz_file['theta_safety']
        self.reward = npz_file['reward']
        self.theta_reward = npz_file['theta_reward']
        self.true_safety_map = npz_file['true_safety_map']


with open(hydra.utils.to_absolute_path('config.yaml')) as file:
    cfg_env = yaml.safe_load(file)['env']


class EmptyEnv_SPOLF(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(
            size=cfg_env['size'],
            view_size=cfg_env['view_size'],
            dim_feature=cfg_env['dim_feature'],
            **kwargs
            )


register(
    id='MiniGrid-Empty-SPOLF-v0',
    entry_point='myenv:EmptyEnv_SPOLF'
)
