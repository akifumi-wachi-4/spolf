# Author: Akifumi Wachi
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

import random
from math import sqrt, log

import numpy as np
import mdptoolbox
from scipy import optimize
import GPy


class SPOLF_Agent(object):
    def __init__(self, env, cfg, h) -> None:

        # Maps
        self.feature_map = np.zeros((env.size, env.size, env.dim_feature))
        self.safety_map = np.zeros((env.size, env.size, 2), dtype=bool)
        self.safety_reachability_map = np.zeros(
            (env.size, env.size, 2), dtype=bool
            )
        self.is_observed_map = np.zeros((env.size, env.size), dtype=bool)
        self.is_visited_map = np.zeros((env.size, env.size), dtype=bool)

        # History of feature, reward, and safety
        self.feature_history = []
        self.reward_history = []
        self.safety_history = []

        # Coefficient for estimating reward and safety functions
        self.theta_hat_reward = np.zeros(env.dim_feature)
        self.theta_hat_safety = np.zeros(env.dim_feature)

        # Observation noise
        self.sigma = env.sigma

        # Safe MDP parameters
        self.gamma = cfg.agent.gamma
        self.P = self._get_transition(env)
        self.pi0 = None
        self.V0 = None
        self.h = h
        self.delta = cfg.agent.delta

        # link function
        self.link = env.link
        self.kappa = env.kappa
        self.L_mu = env.L_mu

        self.bounds_reward = np.zeros((env.size, env.size, 2))
        self.bounds_safety = np.zeros((env.size, env.size, 2))

        self.sim_type = cfg.sim_type

        if 'gp' not in self.sim_type:
            self.assumption_type = 'glm'
            # Upper and lower bounds for reward and safety
            # - bounds_reward, bounds_safety: numpy array
            #   - 0: x_axis,  1: y_axis,  2: {0: upper bound, 1: lower bound}
            self.W = np.identity(env.dim_feature)
            self.weight_norm_inv_W = np.zeros((env.size, env.size))
            self.eig_max_inv_W = 0

            _beta_coef = 3 * self.sigma / self.kappa
            self.beta_reward = _beta_coef * sqrt(log(3/self.delta))
            self.beta_safety = _beta_coef * sqrt(log(3/self.delta))

            if self.sim_type == 'safe_glm_stepwise':
                self.safe_glm_stepwise_phase = 1
                # Migration threshold
                self.glm_omega = cfg.agent.step_glm_migration_threshold

        else:
            self.assumption_type = 'gp'
            self.agent_pos_history = []
            self.gp_reward = None
            self.gp_safety = None
            self.beta_reward = 2
            self.beta_safety = 2
            self.safe_gp_stepwise_phase = 1
            self.uncertainty = np.zeros((env.size, env.size))
            # Migration threshold
            self.gp_omega = cfg.agent.step_gp_migration_threshold

            self.reward_lik = GPy.likelihoods.Gaussian(variance=env.sigma**2)
            self.safety_lik = GPy.likelihoods.Gaussian(variance=env.sigma**2)

            if self.sim_type == 'safe_gp_feature':
                self.kernel_reward = GPy.kern.RBF(input_dim=env.dim_feature)
                self.kernel_safety = GPy.kern.RBF(input_dim=env.dim_feature)

            elif self.sim_type == 'safe_gp_state':
                self.kernel_reward = GPy.kern.RBF(
                    input_dim=2, lengthscale=(2., 2.), ARD=True
                    )
                self.kernel_safety = GPy.kern.RBF(
                    input_dim=2, lengthscale=(2., 2.), ARD=True
                    )
    
    def _within_wall(self, pos, env_size):
        if (pos[0] >= 0) and (pos[0] < env_size):
            if (pos[1] >= 0) and (pos[1] < env_size):
                return True
        return False

    def _get_transition(self, env) -> np.ndarray:
        _n_states = env.size**2
        _P = np.zeros((5, _n_states, _n_states))
        # "GO NORTH / EAST / SOUTH / WEST" actions
        for _a in range(4):
            for i in range(_n_states):
                _next_pos = self._idx2coord(env, i) + env.DIR_TO_VEC[_a]
                if self._within_wall(_next_pos, env.size):
                    _next_cell = env.grid.get(*_next_pos)
                    if (_next_cell is None) or _next_cell.can_overlap():
                        _P[_a, i, self._coord2idx(env, _next_pos)] = 1
                    else:
                        _P[_a, i, i] = 1
                else:
                    _P[_a, i, i] = 1
        # "STAY" action
        for i in range(_n_states):
            _P[4, i, i] = 1
        return _P

    def _coord2idx(self, env, coord) -> int:
        return env.size * coord[0] + coord[1]

    def _idx2coord(self, env, idx) -> np.ndarray:
        return np.array([int(idx / env.size), int(idx % env.size)])

    def initialization(self, env, cfg) -> np.ndarray:
        chosen_states = np.array(
            [random.randint(0, env.size-1), random.randint(0, env.size-1)]
            )

        # feature
        feat = env.feature[chosen_states[0], chosen_states[1]]
        self.feature_history.append(feat)

        # reward
        noise_reward = np.random.normal(loc=0, scale=self.sigma)
        _reward = env.reward[chosen_states[0], chosen_states[1]] + noise_reward
        self.reward_history.append(_reward)

        # safety
        noise_safety = np.random.normal(loc=0, scale=self.sigma)
        _safety = env.safety[chosen_states[0], chosen_states[1]] + noise_safety
        self.safety_history.append(_safety)

        if self.assumption_type == 'glm':
            # GLM parameters
            self.W += np.outer(feat, feat)
            _inv_W = np.linalg.inv(self.W)
            self.eig_max_inv_W = max(np.linalg.eig(_inv_W)[0])
            self.update_coefficient()

        elif self.assumption_type == 'gp':
            self.agent_pos_history.append([chosen_states[0], chosen_states[1]])
            if len(self.reward_history) == cfg.agent.num_init_info:
                n_samples = cfg.agent.num_init_info
                # GP parameters
                if self.sim_type == 'safe_gp_feature':
                    self.gp_reward = GPy.core.GP(
                        np.array(self.feature_history),
                        np.array(self.reward_history).reshape(n_samples, 1),
                        self.kernel_reward, self.reward_lik
                        )
                    self.gp_safety = GPy.core.GP(
                        np.array(self.feature_history),
                        np.array(self.safety_history).reshape(n_samples, 1),
                        self.kernel_safety, self.safety_lik
                        )
                elif self.sim_type == 'safe_gp_state':
                    self.gp_reward = GPy.core.GP(
                        np.array(self.feature_history),
                        np.array(self.reward_history).reshape(n_samples, 1),
                        self.kernel_reward, self.reward_lik
                        )
                    self.gp_safety = GPy.core.GP(
                        np.array(self.feature_history),
                        np.array(self.safety_history).reshape(n_samples, 1),
                        self.kernel_safety, self.safety_lik
                        )

        else:
            NotImplementedError

        return chosen_states

    def update(self, env, cfg, obs, feature, reward, safety) -> None:
        self.update_basemap(env, cfg, obs, feature, reward, safety)
        self.update_history(env, reward, safety)

        if self.assumption_type == 'glm':
            self.update_coefficient()
            self.update_glm_bounds(env)
        else:
            if self.sim_type == 'safe_gp_feature':
                self.update_gp_bounds_feature(env)
            else:
                self.update_gp_bounds_state(env)

        self.update_safemap(env)

    def update_basemap(self, env, cfg, obs, feature, reward, safety) -> None:
        # Update the "visited" map
        self.is_visited_map[env.agent_pos[0], env.agent_pos[1]] = True

        if cfg.env.hyp_obs:
            # Update the "observed map"
            obs_s = feature['observed_states']
            for _o in obs_s:
                self.is_observed_map[_o[0], _o[1]] = True
            # Update the "feature map"
            obs_feature = feature['feature']
            for i in range(len(obs_s)):
                self.feature_map[obs_s[i][0], obs_s[i][1]] = obs_feature[i]
        else:
            for i in range(env.size):
                for j in range(env.size):
                    if self.is_visited_map[i, j]:
                        self.is_observed_map[i, j] = True
                        self.feature_map[i, j] = env.feature[i, j]
            # Add feature in the front
            fwd_pos = [env.front_pos[0], env.front_pos[1]]
            if fwd_pos[0] > 0 and fwd_pos[0] < env.size-1:
                if fwd_pos[1] > 0 and fwd_pos[1] < env.size-1:
                    self.is_observed_map[fwd_pos[0], fwd_pos[1]] = True
                    _feature = env.feature[fwd_pos[0], fwd_pos[1]]
                    self.feature_map[fwd_pos[0], fwd_pos[1]] = _feature

    def update_history(self, env, reward, safety) -> None:
        feat = env.feature[env.agent_pos[0], env.agent_pos[1]]
        self.feature_history.append(feat)
        self.reward_history.append(reward)
        self.safety_history.append(safety)

    def update_glm_bounds(self, env) -> None:
        # Update matrix W
        feat_agent_pos = env.feature[env.agent_pos[0], env.agent_pos[1]]
        self.W += np.outer(feat_agent_pos, feat_agent_pos)
        _inv_W = np.linalg.inv(self.W)

        # Minimum eigen value of W
        # eig_min_W = min(np.linalg.eig(self.W)[0])
        # Maximum eigen value of W^{-1}
        self.eig_max_inv_W = max(np.linalg.eig(_inv_W)[0])

        for i in range(env.size):
            for j in range(env.size):
                if self.is_observed_map[i, j]:
                    feat = env.feature[i, j]
                    # Expected reward is defined as \mu( c \cdot \hat{\theta} )
                    _mean_reward = self.link(
                        np.dot(feat.T, self.theta_hat_reward)
                        )
                    _mean_safety = self.link(
                        np.dot(feat.T, self.theta_hat_safety)
                        )

                    # Uncertainty term is defined as \beta \|c\|_{W^{-1}}
                    self.weight_norm_inv_W[i, j] = sqrt(
                            np.dot(np.dot(feat.T, _inv_W), feat)
                            )
                    _W = self.weight_norm_inv_W[i, j]
                    _uncertainty_reward = self.L_mu * self.beta_reward * _W
                    _uncertainty_safety = self.L_mu * self.beta_safety * _W

                else:
                    # Maximum reward is defined as \mu( \|c\|_2 )
                    _mean_reward = max(self.link(
                            np.linalg.norm(self.theta_hat_reward, ord=2)
                            ), 1)
                    _mean_safety = max(self.link(
                            np.linalg.norm(self.theta_hat_safety, ord=2)
                            ), 1)

                    # Maximum uncertainty \beta sqrt{\eig_max(W^{-1})}
                    _W = sqrt(self.eig_max_inv_W)
                    _uncertainty_reward = self.L_mu * self.beta_reward * _W
                    _uncertainty_safety = self.L_mu * self.beta_safety * _W

                # Upper bound for reward and safety functions
                self.bounds_reward[i, j, 0] = _mean_reward + _uncertainty_reward
                self.bounds_safety[i, j, 0] = _mean_safety + _uncertainty_safety

                # Lower bound for safety function
                if self.is_observed_map[i, j]:
                    self.bounds_reward[i, j, 1] = _mean_reward - _uncertainty_reward
                    self.bounds_safety[i, j, 1] = _mean_safety - _uncertainty_safety
                else:
                    self.bounds_reward[i, j, 1] = 0
                    self.bounds_safety[i, j, 1] = 0


    def update_gp_bounds_feature(self, env) -> None:
        # Update GP observations
        n_samples = len(self.reward_history)
        self.gp_reward.set_XY(
            np.array(self.feature_history),
            np.array(self.reward_history).reshape(n_samples, 1)
            )
        self.gp_safety.set_XY(
            np.array(self.feature_history),
            np.array(self.safety_history).reshape(n_samples, 1)
            )

        for i in range(env.size):
            for j in range(env.size):
                if self.is_observed_map[i, j]:
                    feat = env.feature[i, j].reshape(1, env.dim_feature)
                    mu_r = self.gp_reward.predict(feat)[0]
                    std_r = sqrt(self.gp_reward.predict(feat)[1])
                    mu_g = self.gp_safety.predict(feat)[0]
                    std_g = sqrt(self.gp_safety.predict(feat)[1])

                    # Upper bounds
                    self.bounds_reward[i, j, 0] = mu_r + self.beta_reward * std_r
                    self.bounds_safety[i, j, 0] = mu_g + self.beta_safety * std_g
                    # Lower bounds
                    self.bounds_reward[i, j, 1] = mu_r - self.beta_reward * std_r
                    self.bounds_safety[i, j, 1] = mu_g - self.beta_safety * std_g

                    self.uncertainty[i, j] = std_g

                else:
                    # Upper bounds
                    self.bounds_reward[i, j, 0] = 1
                    self.bounds_safety[i, j, 0] = 1
                    # Lower bounds
                    self.bounds_reward[i, j, 1] = 0
                    self.bounds_safety[i, j, 1] = 0

    def update_gp_bounds_state(self, env) -> None:
        # Update GP observations
        n_samples = len(self.reward_history)
        self.agent_pos_history.append([env.agent_pos[0], env.agent_pos[1]])
        self.gp_reward.set_XY(
                np.array(self.agent_pos_history),
                np.array(self.reward_history).reshape(n_samples, 1)
                )
        self.gp_safety.set_XY(
                np.array(self.agent_pos_history),
                np.array(self.safety_history).reshape(n_samples, 1)
                )

        for i in range(env.size):
            for j in range(env.size):
                _pos = np.array([i, j]).reshape(1, 2)
                mu_r = self.gp_reward.predict(_pos)[0]
                std_r = sqrt(self.gp_reward.predict(_pos)[1])
                mu_g = self.gp_safety.predict(_pos)[0]
                std_g = sqrt(self.gp_safety.predict(_pos)[1])

                # Upper bounds
                self.bounds_reward[i, j, 0] = mu_r + self.beta_reward * std_r
                self.bounds_safety[i, j, 0] = mu_g + self.beta_safety * std_g
                # Lower bounds
                self.bounds_reward[i, j, 1] = mu_r - self.beta_reward * std_r
                self.bounds_safety[i, j, 1] = mu_g - self.beta_safety * std_g

                self.uncertainty[i, j] = std_g

    def _safety_reachablity(self, env, initial_nodes) -> np.ndarray:
        checked = np.zeros((env.size, env.size, 2), dtype=bool)
        checked[initial_nodes[0], initial_nodes[1], :] = True

        for i in range(2):
            stack = []
            stack.append(initial_nodes)

            while stack:
                node = stack.pop(0)
                adjacent = node + np.array(env.DIR_TO_VEC)
                for adj in adjacent:
                    try:
                        _checked = checked[adj[0], adj[1], i]
                        _safe = self.safety_map[adj[0], adj[1], i]
                        if not _checked and _safe:
                            checked[adj[0], adj[1], i] = True
                            stack.append(adj)
                    except Exception as e:
                        print('reachability', e, adj)
                        continue
        return checked

    def update_safemap(self, env) -> None:
        # Update (binary) safety map
        for i in range(env.size):
            for j in range(env.size):
                if self.is_observed_map[i, j]:
                    self.safety_map[i, j, 0] = (
                            self.bounds_safety[i, j, 0] >= self.h
                            )
                    self.safety_map[i, j, 1] = (
                            self.bounds_safety[i, j, 1] >= self.h
                            )
                else:
                    self.safety_map[i, j, 0] = (
                            self.bounds_safety[i, j, 0] >= self.h
                            )

                if env.is_wall(i, j):
                    self.safety_map[i, j, 0] = False

        _start_pos = np.array([env.agent_start_pos[0], env.agent_start_pos[1]])
        self.safety_reachability_map = self._safety_reachablity(env, _start_pos)

    def update_coefficient(self) -> None:
        # Optimize the coefficients based on the data
        _theta_hat_reward = optimize.root(
                self.__to_optimize,
                self.theta_hat_reward,
                self.reward_history
                ).x
        _theta_hat_safety = optimize.root(
                self.__to_optimize,
                self.theta_hat_safety,
                self.safety_history
                ).x

        # Update the coefficients
        self.theta_hat_reward = _theta_hat_reward
        self.theta_hat_safety = _theta_hat_safety

    def __to_optimize(self, theta, func_history) -> float:
        to_sum = []
        for t in range(len(func_history)):
            _feat = self.feature_history[t]
            to_sum.append((
                func_history[t] - self.link(np.inner(_feat, theta))
                ) * _feat)
        return np.sum(to_sum, 0)

    def _get_neighbor_states(self, env) -> list:
        neighbor_states = []
        for i in range(4):
            _next_pos = env.agent_pos + env.DIR_TO_VEC[i]
            _next_cell = env.grid.get(*_next_pos)

            if (_next_cell is None) or _next_cell.can_overlap():
                neighbor_states.append(_next_pos)
        return neighbor_states

    def get_next_state(self, env, V, pi, agent_type) -> list:
        # Safe agent chooses the next state with in the pessmistic safe space
        if agent_type == 'safe_glm':
            adjacent = self._get_neighbor_states(env)
            list_V = []
            safe_adjacent = []
            for adj in adjacent:
                try:
                    if self.safety_reachability_map[adj[0], adj[1], 1]:
                        safe_adjacent.append(adj)
                        list_V.append(V[self._coord2idx(env, adj)])
                except:
                    print('error', adj)
                    continue
            safe_adjacent.append(env.agent_pos)
            list_V.append(V[self._coord2idx(env, env.agent_pos)])
            _idx_maxV = np.argmax(list_V)
            next_state = safe_adjacent[_idx_maxV]

        # Unsafe or oracle agents simply choose the next state based
        # on the value function
        else:
            _action = pi[self._coord2idx(env, env.agent_pos)]
            idx_next_state = np.where(
                    self.P[_action, self._coord2idx(env, env.agent_pos), :]
                    )[0][0]
            next_state = self._idx2coord(env, idx_next_state)

        return next_state

    def get_reward(self, env, agent_type, stack_workaround=None) -> np.ndarray:
        # Safe GLM agent
        if agent_type == 'safe_glm':
            _reward = np.zeros((env.size, env.size))
            for i in range(env.size):
                for j in range(env.size):
                    if stack_workaround == 'pess_optim':
                        _reward[i, j] = self.bounds_reward[i, j, 0]
                        _penalty = not self.safety_reachability_map[i, j, 1]
                    elif stack_workaround == 'etse':
                        _reward[i, j] = self.weight_norm_inv_W[i, j]
                        _penalty = not self.safety_reachability_map[i, j, 1]
                    else:
                        _reward[i, j] = self.bounds_reward[i, j, 0]
                        _penalty = not self.safety_reachability_map[i, j, 0]

                    if _penalty:
                        _reward[i, j] = -1e5

            R = np.reshape(_reward[:, :], env.size**2)

        # Step-wise Safe GLM agent
        elif agent_type == 'safe_glm_stepwise':
            _reward = np.zeros((env.size, env.size))
            for i in range(env.size):
                for j in range(env.size):
                    if self.safe_glm_stepwise_phase == 1:
                        _reward[i, j] = self.weight_norm_inv_W[i, j]
                    else:
                        _reward[i, j] = self.bounds_reward[i, j, 0]
                    _penalty = not self.safety_reachability_map[i, j, 1]

                    if _penalty:
                        _reward[i, j] = -1e5

            R = np.reshape(_reward[:, :], env.size**2)

        # Safety-agnostic agent
        elif agent_type == 'unsafe_glm':
            R = np.reshape(self.bounds_reward[:, :, 0], env.size**2)

        # Oracle agent
        elif agent_type == 'oracle':
            _reward = np.zeros((env.size, env.size))
            for i in range(env.size):
                for j in range(env.size):
                    _reward[i, j] = env.reward[i, j]
                    if not env.true_safety_map[i, j]:
                        _reward[i, j] = -1e5
            R = np.reshape(_reward[:, :], env.size**2)

        # Step-wise Safe GP-based contexutual agent
        elif agent_type == 'safe_gp_feature' or agent_type == 'safe_gp_state':
            _reward = np.zeros((env.size, env.size))
            for i in range(env.size):
                for j in range(env.size):
                    if self.safe_gp_stepwise_phase == 1:
                        _reward[i, j] = self.uncertainty[i, j]
                    else:
                        _reward[i, j] = self.bounds_reward[i, j, 0]
                    _penalty = not self.safety_reachability_map[i, j, 1]

                    if _penalty:
                        _reward[i, j] = -1e5

            R = np.reshape(_reward[:, :], env.size**2)

        else:
            R = None
            NotImplementedError

        for i in range(env.size**2):
            R[i] = min(R[i], 1)

        return R

    def chck_step_migration(self, env) -> None:
        if self.sim_type == 'safe_glm_stepwise':
            list_weight_norm_inv_W = []
            for i in range(env.size):
                for j in range(env.size):
                    if self.safety_reachability_map[i, j, 1]:
                        list_weight_norm_inv_W.append(
                                self.weight_norm_inv_W[i, j]
                                )
            if len(list_weight_norm_inv_W) > 1:
                _max_norm_inv_W = max(list_weight_norm_inv_W)
                if _max_norm_inv_W < self.glm_omega:
                    print("Phase is migrated to Step 2.")
                    self.safe_glm_stepwise_phase = 2
        else:
            list_uncertainty_term = []
            for i in range(env.size):
                for j in range(env.size):
                    if self.safety_reachability_map[i, j, 1]:
                        list_uncertainty_term.append(self.uncertainty[i, j])
            if len(list_uncertainty_term) > 1:
                _max_uncertainty = max(list_uncertainty_term)
                if _max_uncertainty < self.gp_omega:
                    print("Phase is migrated to Step 2.")
                    self.safe_gp_stepwise_phase = 2

    def chck_contradiction(self, env, next_state) -> bool:
        tst_next_action = self.pi0[self._coord2idx(env, env.agent_pos)]
        _P = self.P[tst_next_action, self._coord2idx(env, env.agent_pos), :]
        tst_next_state = np.where(_P)
        if abs(self._coord2idx(env, next_state) - tst_next_state[0]) != 0:
            return True
        else:
            return False

    def optimize_policy(
            self, env, cfg, stack_workaround=None, update_policy=True
            ) -> np.ndarray:
        if update_policy:
            R = self.get_reward(env, cfg.sim_type, stack_workaround)
            if cfg.agent.rl_algo == 'PI':
                y = mdptoolbox.mdp.PolicyIteration(
                        self.P, R, self.gamma, policy0=self.pi0, max_iter=100
                        )
            elif cfg.agent.rl_algo == 'AMPI':
                y = mdptoolbox.mdp.PolicyIterationModified(
                        self.P, R, self.gamma, max_iter=100
                        )
            else:
                y = None
                NotImplementedError
            y.run()
            V, pi = y.V, y.policy
            if stack_workaround is None:
                self.V0, self.pi0 = V, pi
        else:
            V, pi = self.V0, self.pi0

        next_state = self.get_next_state(env, V, pi, cfg.sim_type)

        if cfg.sim_type == 'safe_glm_stepwise':
            if self.safe_glm_stepwise_phase == 1:
                self.chck_step_migration(env)

        if cfg.sim_type == 'safe_gp_feature':
            if self.safe_gp_stepwise_phase == 1:
                self.chck_step_migration(env)

        return next_state


class SPOLF_Agent_Safety_Gym(SPOLF_Agent):
    def __init__(self, env, cfg, h) -> None:
        super().__init__(env, cfg, h)
        self.env = env

    def _get_transition(self, env) -> np.ndarray:
        _n_states = env.size**2
        _P = np.zeros((5, _n_states, _n_states))
        # "GO NORTH/EAST/SOUTH/WEST" actions
        for _a in range(4):
            for i in range(_n_states):
                _next_pos = self._idx2coord(env, i) + env.DIR_TO_VEC[_a]
                if self._within_wall(_next_pos, env.size):
                    _P[_a, i, self._coord2idx(env, _next_pos)] = 1
                else:
                    _P[_a, i, i] = 1
        # "STAY" action
        for i in range(_n_states):
            _P[4, i, i] = 1
        return _P

    def __to_optimize(self, theta, func_history) -> float:
        to_sum = []
        for t in range(len(func_history)):
            _feat = self.feature_history[t]
            to_sum.append((
                func_history[t] - self.link(np.inner(_feat, theta))
                ) * _feat)
        return np.sum(to_sum, 0)

    def initialization_safety_gym(self, env) -> None:
        # feature
        obs = env.obs()

        # only use goal and hazard lidar
        feat = np.r_[
            np.max(obs['goal_lidar']),
            np.max(obs['hazards_lidar']),
            1
        ]
        self.feature_history.append(feat)

        # reward
        act = [0., 0.]
        obs, reward, done, info = env.step(act)
        cost = (1 - env.cost()['cost'])
        self.reward_history.append(reward)

        # safety
        self.safety_history.append(cost)

        if self.assumption_type == 'glm':
            # GLM parameters
            self.W += np.outer(feat, feat)
            _inv_W = np.linalg.inv(self.W)
            self.eig_max_inv_W = max(np.linalg.eig(_inv_W)[0])
            self.update_coefficient()

        else:
            NotImplementedError

    def _get_neighbor_states(self, env) -> list:
        neighbor_states = []
        for i in range(4):
            _next_pos = env.agent_pos + env.DIR_TO_VEC[i]
            if self._within_wall(_next_pos, env.size):
                neighbor_states.append(_next_pos)
        return neighbor_states

    def update_coefficient(self) -> None:
        theta_hat_reward = np.zeros(self.env.dim_feature)
        theta_hat_safety = np.zeros(self.env.dim_feature)
        try:
            r_reward = optimize.root(
                self.__to_optimize,
                theta_hat_reward,
                self.reward_history,
                method='hybr'
                )
        except Exception as e:
            print('update reward', e)

        try:
            r_safety = optimize.root(
                self.__to_optimize,
                theta_hat_safety,
                self.safety_history,
                method='hybr'
                )
        except Exception as e:
            print('update safety', e)
            return

        if r_reward['success']:
            self.safety_update_num = len(self.safety_history)
        self.theta_hat_reward = r_reward.x
        if r_safety['success']:
            self.reward_update_num = len(self.reward_history)
        self.theta_hat_safety = r_safety.x
