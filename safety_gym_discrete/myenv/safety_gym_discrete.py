# Author: Akifumi Wachi, Yunyue Wei
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import time
from copy import deepcopy

import numpy as np

from safety_gym.envs.engine import Engine, ResamplingError


def coord_to_safety_gym_pos(coord, world_shape, step_size):
    """Convert coord to safety-gym position"""
    MAP_SIZE = 7 * world_shape[0] / 20
    _ax = MAP_SIZE / world_shape[0]
    _ay = MAP_SIZE / world_shape[1]
    _pos_x = _ax * (coord[0] / step_size[0]) - MAP_SIZE / 2 + _ax / 2
    _pos_y = _ay * (coord[1] / step_size[1]) - MAP_SIZE / 2 + _ay / 2

    return np.array([_pos_x, _pos_y])

def safety_gym_pos_to_coord(pos, world_shape, step_size):
    """Convert safety-gym position to coord"""
    MAP_SIZE = 7 * world_shape[0] / 20
    _pos_x = pos[0]
    _pos_y = pos[1]
    _ax = MAP_SIZE / world_shape[0]
    _ay = MAP_SIZE / world_shape[1]
    _coordx = (_pos_x - _ax / 2 + MAP_SIZE / 2) / _ax * step_size[0]
    _coordy = (_pos_y - _ay / 2 + MAP_SIZE / 2) / _ay * step_size[1]
    _round_coordx = np.round(_coordx / step_size[0]) * step_size[0]
    _round_coordy = np.round(_coordy / step_size[1]) * step_size[1]

    return np.array([_round_coordx, _round_coordy])


class Engine_Discrete(Engine):
    def __init__(
        self,
        config,
        world_shape=(5, 5),
        view_size=9,
        step_size=(1, 1),
        link=lambda x: x,
        kappa=1,
        L_mu=1,
        sigma=0.01,
        h=0.7
        ):
        self.DEFAULT['allow_conflict'] = False
        super(Engine_Discrete, self).__init__(config)

        # if allow_conflict,
        # a fake env can be created to get the feature vector
        if 'allow_conflict' in config.keys():
            self.allow_conflict = config['allow_conflict']

        n, m = world_shape
        step1, step2 = step_size
        xx, yy = np.meshgrid(
            np.linspace(0, (n - 1) * step1, n),
            np.linspace(0, (m - 1) * step2, m),
            indexing="ij"
            )
        self.coord = np.vstack((xx.flatten(), yy.flatten())).T
        self.reset()
        self.config = config
        # safety-gym world
        self.world_shape = world_shape
        self.MAP_SIZE = 7 * world_shape[0] / 20
        self.a = self.MAP_SIZE / world_shape[0]
        self.view_size = view_size
        self.step_size = step_size
        self.size = self.world_shape[0]
        # Link function
        self.link = link
        self.kappa = kappa
        self.L_mu = L_mu
        # agent hyperparameter
        self.dim_feature = 3
        self.h = h
        self.sigma = sigma


        robot_pos = safety_gym_pos_to_coord(
            self.robot_pos,
            self.world_shape,
            self.step_size
        )
        robot_posx = int(robot_pos[0] / self.step_size[0])
        robot_posy = int(robot_pos[1] / self.step_size[1])

        self.agent_start_pos = (robot_posx, robot_posy)
        self.agent_pos = (robot_posx, robot_posy)

        self.DIR_TO_VEC = [
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
            # Up (negative Y)
            np.array((0, -1))
            ]

    def sample_layout(self):
        '''
        Sample a single layout,
        returning True if successful, else False.
        '''
        def placement_is_valid(xy, layout):
            if self.allow_conflict:
                return True
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(xy - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            return True

        layout = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                xy = self.draw_placement(placements, keepout)
                if placement_is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        self.layout = layout
        return True

    def build_goal_position(self):
        ''' Build a new goal position, maybe with resampling due to hazards '''
        # Resample until goal is compatible with layout
        if self.allow_conflict == False:
            if 'goal' in self.layout:
                del self.layout['goal']
            for _ in range(10000):  # Retries
                if self.sample_goal_position():
                    break
            else:
                raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        self.world_config_dict['geoms']['goal']['pos'][:2] = self.layout['goal']
        goal_body_id = self.sim.model.body_name2id('goal')
        self.sim.model.body_pos[goal_body_id][:2] = self.layout['goal']
        self.sim.forward()

    def reset(self):
        ''' Reset the physics simulation and return observation '''
        self._seed += 1  # Increment seed
        self.rs = np.random.RandomState(self._seed)
        self.done = False
        self.steps = 0  # Count of steps taken in this episode
        # Set the button timer to zero (so button is immediately visible)
        self.buttons_timer = 0

        self.clear()
        self.build()
        # Save the layout at reset
        self.reset_layout = deepcopy(self.layout)

        cost = self.cost()
        if self.allow_conflict == False:
            assert cost['cost'] == 0, f'World has starting cost! {cost}'

        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        return self.obs()

    def discreate_step(self, pos, render=False):
        """ Move to the specified discretized position """

        # Adjust the direction to the target position
        grid_pos = pos
        pos = coord_to_safety_gym_pos(pos, self.world_shape, self.step_size)
        discrete_reward = 0.0
        total_step = 0
        total_cost = 0
        st = time.time()
        act = [0, 0]
        assert self.action_space.contains(act)
        obs, reward, done, info = self.step(act)

        while abs(self.ego_xy(pos)[1]) > 5e-3 or self.ego_xy(pos)[0] < 0:
            total_step += 1
            act = [0, 0.15]
            assert self.action_space.contains(act)
            obs, reward, done, info = self.step(act)
            discrete_reward += reward
            total_cost += info['cost']

            self.done = False
            assert self.observation_space.contains(obs)
            if render:
                self.render()
        print('turn using', st - time.time())

        st = time.time()
        # Adjust the distance to the specified position
        while abs(self.ego_xy(pos)[0]) > 5e-3:
            total_step += 1
            #avoid too fast speed
            if np.linalg.norm(obs['velocimeter']) > 0.08:
                act = [0.0, 0]
            else:
                act = [0.005, 0]
            
            assert self.action_space.contains(act)
            obs, reward, done, info = self.step(act)
            total_cost += info['cost']
            discrete_reward += reward
            self.done = False
            assert self.observation_space.contains(obs)
            if render:
                self.render()
        print('go using', st - time.time())

        feature = {'observed_states': [], 'feature': []}
        vision_pos = self.get_view_sets()
        for g_pos in vision_pos:
            feature['observed_states'].append(
                [int(g_pos[0] / self.step_size[0]),
                 int(g_pos[1] / self.step_size[1])]
                )
            feature['feature'].append(
                self.feature[int(g_pos[0] / self.step_size[0]),
                             int(g_pos[1] / self.step_size[1])]
                )
        act = [0, 0]
        assert self.action_space.contains(act)
        obs, reward, done, info = self.step(act)
        cost = self.safety[grid_pos[0], grid_pos[1]]
        robot_pos = safety_gym_pos_to_coord(
            self.robot_pos,
            self.world_shape,
            self.step_size
        )
        robot_posx = int(robot_pos[0] / self.step_size[0])
        robot_posy = int(robot_pos[1] / self.step_size[1])
        self.agent_pos = (robot_posx, robot_posy)
        if not self.goal_met():
            reward += discrete_reward

        self.steps = 1 # avoid reaching safety-gym step limit

        return self.obs(), feature, cost, reward, total_step, total_cost

    def get_view_sets(self):
        """Get the square set of tiles visible to the agent"""
        assert self.view_size % 2 == 1
        cur_grid_pos = safety_gym_pos_to_coord(
            self.robot_pos,
            self.world_shape,
            self.step_size
        )
        view_pos = []

        for i in range(-(self.view_size // 2), self.view_size // 2 + 1):
            for j in range(-(self.view_size // 2), self.view_size // 2 + 1):
                extend_grid_pos = [cur_grid_pos[0] + i * self.step_size[0],
                                   cur_grid_pos[1] + j * self.step_size[1]]
                if 0 <= extend_grid_pos[0] <= np.max(self.coord) \
                        and 0 <= extend_grid_pos[1] <= np.max(self.coord):
                    view_pos.append(extend_grid_pos)
        return view_pos

    def get_info(self):
        """
        Get safety function value, reward function value and feature value
        by creating a "fake" environment,
        where the hazard is the same as actualenvironment.
        We put the agent in every discretized position and get values.
        """
        self.feature = np.random.rand(
            self.world_shape[0],
            self.world_shape[1],
            self.dim_feature
            )
        self.safety = np.random.rand(
            self.world_shape[0],
            self.world_shape[1]
            )
        self.reward_fun = np.random.rand(
            self.world_shape[0],
            self.world_shape[1]
            )
        for i in range(self.world_shape[0]):
            for j in range(self.world_shape[1]):
                index = i * self.world_shape[0] + j
                print(self.coord[index])
                pos = coord_to_safety_gym_pos(
                    self.coord[index], self.world_shape, self.step_size
                    )
                fake_config = self.config.copy()

                # set same goal and hazard pos
                fake_config['hazards_locations'] = [x.tolist()[:2]
                                                    for x in self.hazards_pos]
                fake_config['goal_locations'] = [self.goal_pos.tolist()[:2]]
                fake_config['allow_conflict'] = True

                # set fake pos
                fake_config['robot_locations'] = [pos.tolist()]
                fake_config['robot_rot'] = self.robot_rot
                try:
                    fake_env = Engine_Discrete(fake_config, self.world_shape)
                    fake_env.reset()
                    # fake_env.render()
                    act = [0., 0.]
                    obs, reward, done, info = fake_env.step(act)
                    cost = (1 - fake_env.cost()['cost'])
                    assert cost == 1 or cost == 0
                    c = np.r_[
                        np.max(obs['goal_lidar']),
                        np.max(obs['hazards_lidar']),
                        1]
                    print('goal', np.max(obs['goal_lidar']))
                    print('hazard', np.max(obs['hazards_lidar']))
                    # only use goal and hazard lidar
                    self.feature[i, j] = c
                    self.safety[i, j] = cost
                    self.reward_fun[i, j] = reward
                    print(obs['goal_lidar'])
                    print(obs['hazards_lidar'])
                    print(reward)
                    print(cost)
                    del fake_env, fake_config
                except Exception as e:
                    print(pos, e)
                    continue


    def _safety_reachablity(self, safety_map, initial_nodes) -> np.ndarray:
        checked = np.zeros((self.size, self.size), dtype=bool)
        checked[initial_nodes[0], initial_nodes[1]] = True
        stack = []
        stack.append(initial_nodes)
        while stack:
            node = stack.pop(0)
            adjacent = node + np.array(self.DIR_TO_VEC)
            for adj in adjacent:
                try:
                    if not checked[adj[0], adj[1]] \
                            and safety_map[adj[0], adj[1]]:
                        checked[adj[0], adj[1]] = True
                        stack.append(adj)
                except Exception as e:
                    print(e)
                    continue

        return checked

    def is_wall(self, i, j) -> bool:
        if i == -1 or i == self.size or j == -1 or j == self.size:
            return True
        else:
            return False



