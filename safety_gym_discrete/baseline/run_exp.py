# Author: Akifumi Wachi, Yunyue Wei
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import time
import numpy as np
import tensorflow as tf
import gym

import safe_rl.pg.trust_region as tro
from safe_rl.pg.agents import PPOAgent, TRPOAgent, CPOAgent
from safe_rl.pg.buffer import CPOBuffer
from safe_rl.pg.network import count_vars, \
    get_vars, \
    mlp_actor_critic, \
    placeholders, \
    placeholders_from_spaces
from safe_rl.pg.utils import values_as_sorted_list
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from safe_rl.utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum
from safety_gym.envs.engine import Engine

"""
A modification to run_agent.py in safety-starter-agent repository, 
to add test environment in the experiment.
"""

def coord_to_safety_gym_pos(coord, world_shape, step_size=(1, 1)):
    MAP_SIZE = 7 * world_shape[0] / 20

    _ax = MAP_SIZE / world_shape[0]
    _ay = MAP_SIZE / world_shape[1]
    MAP_SIZE = 7 * world_shape[0] / 20
    _pos_x = _ax * (coord[0] / step_size[0]) - MAP_SIZE / 2 + _ax / 2
    _pos_y = _ay * (coord[1] / step_size[1]) - MAP_SIZE / 2 + _ay / 2
    return np.array([_pos_x, _pos_y])


# coord_to_safety_gym_pos([5,5], world_shape,step_size)

def safety_gym_pos_to_coord(pos, world_shape, step_size=(1, 1)):
    _pos_x = pos[0]
    _pos_y = pos[1]
    MAP_SIZE = 7 * world_shape[0] / 20

    _ax = MAP_SIZE / world_shape[0]
    _ay = MAP_SIZE / world_shape[1]
    _coordx = (_pos_x - _ax / 2 + MAP_SIZE / 2) / _ax * step_size[0]
    _coordy = (_pos_y - _ay / 2 + MAP_SIZE / 2) / _ay * step_size[1]

    _round_coordx = np.round(_coordx / step_size[0]) * step_size[0]
    _round_coordy = np.round(_coordy / step_size[1]) * step_size[1]

    return np.array([_round_coordx, _round_coordy])


# Multi-purpose agent runner for policy optimization algos
# (PPO, TRPO, their primal-dual equivalents, CPO)

class SafetyGymExp:
    def __init__(
        self,
        config=None,
        world_shape=None,
        agent=PPOAgent(),
        actor_critic=mlp_actor_critic,
        ac_kwargs=dict(),
        seed=0,
        render=False,
        # Experience collection:
        steps_per_epoch=40000,
        # Discount factors:
        gamma=0.99,
        lam=0.97,
        cost_gamma=0.99,
        cost_lam=0.97,
        # Policy learning:
        ent_reg=0.,
        # Cost constraints / penalties:
        cost_lim=25,
        penalty_init=1.,
        penalty_lr=5e-2,
        # KL divergence:
        target_kl=0.01,
        # Value learning:
        vf_lr=1e-3,
        vf_iters=80,
        # Logging:
        logger=None,
        logger_kwargs=dict(),
        ):
        self.initialize_variable(
            agent, config, world_shape,
            ent_reg, target_kl, cost_lim,
            penalty_init, render, seed, vf_iters
            )
        self.prepare_logger(logger, logger_kwargs)
        self.load_env(config)
        self.prepare_AC_graph(ac_kwargs, actor_critic)
        self.create_replay_buffer(
            steps_per_epoch, gamma,
            lam, cost_gamma, cost_lam
            )
        self.create_peanalty_learning_graph(penalty_lr)
        self.create_policy_learning_graph()
        self.create_value_learning_graph(vf_lr)
        self.creat_session()


    def initialize_variable(
        self, agent, config, world_shape,
        ent_reg, target_kl, cost_lim,
        penalty_init, render, seed, vf_iters
        ):
        self.agent = agent
        self.config = config
        self.world_shape = world_shape
        self.ent_reg = ent_reg
        self.target_kl = target_kl
        self.cost_lim = cost_lim
        self.penalty_init = penalty_init
        self.render = render
        self.seed = seed
        self.vf_iters = vf_iters

    def prepare_logger(self, logger,logger_kwargs):
        self.logger = EpochLogger(**logger_kwargs) if logger is None \
            else logger
        self.logger.save_config(locals())
        self.agent.set_logger(self.logger)

    def load_env(self, config=None):
        assert config
        self.env = Engine(config)

    def prepare_AC_graph(self, ac_kwargs, actor_critic):
        # Share information about action space with policy architecture
        self.ac_kwargs = ac_kwargs
        self.ac_kwargs['action_space'] = self.env.action_space

        # Inputs to computation graph from environment spaces
        self.x_ph, self.a_ph = \
            placeholders_from_spaces(self.env.observation_space,
                                     self.env.action_space)

        # Inputs to computation graph for batch data
        self.adv_ph, self.cadv_ph, self.ret_ph, self.cret_ph, self.logp_old_ph\
            = placeholders(*(None for _ in range(5)))

        # Inputs to computation graph for special purposes
        self.surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
        self.cur_cost_ph = tf.placeholder(tf.float32, shape=())

        # Outputs from actor critic
        self.ac_outs = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)
        self.pi, self.logp, self.logp_pi, self.pi_info, \
        self.pi_info_phs, self.d_kl, self.ent, self.v, self.vc = self.ac_outs

        # Organize placeholders for zipping with data from buffer on updates
        self.buf_phs = [self.x_ph, self.a_ph,
                        self.adv_ph, self.cadv_ph,
                        self.ret_ph, self.cret_ph, self.logp_old_ph]
        self.buf_phs += values_as_sorted_list(self.pi_info_phs)

        # Organize symbols we have to compute at each step of acting in env
        self.get_action_ops = dict(pi=self.pi,
                                   v=self.v,
                                   logp_pi=self.logp_pi,
                                   pi_info=self.pi_info)

        # If agent is reward penalized, it doesn't use a separate value function
        # for costs and we don't need to include it in get_action_ops; otherwise we do.
        if not (self.agent.reward_penalized):
            self.get_action_ops['vc'] = self.vc

        # Count variables
        var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
        self.logger.log('\nNumber of parameters: \t pi: %d, '
                        '\t v: %d, \t vc: %d\n' % var_counts)

        # Make a sample estimate for entropy to use as sanity check
        self.approx_ent = tf.reduce_mean(-self.logp)

    def create_replay_buffer(
        self, steps_per_epoch, gamma, lam, cost_gamma, cost_lam
        ):
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        # Obs/act shapes
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        # Experience buffer
        self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        self.pi_info_shapes = {k: v.shape.as_list()[1:]
                               for k, v in self.pi_info_phs.items()}
        self.buf = CPOBuffer(self.local_steps_per_epoch,
                             self.obs_shape,
                             self.act_shape,
                             self.pi_info_shapes,
                             self.gamma,
                             self.lam,
                             self.cost_gamma,
                             self.cost_lam)

    def create_peanalty_learning_graph(self, penalty_lr):
        if self.agent.use_penalty:
            with tf.variable_scope('penalty'):
                # param_init = np.log(penalty_init)
                param_init = np.log(max(np.exp(self.penalty_init) - 1, 1e-8))
                self.penalty_param = \
                    tf.get_variable('penalty_param',
                                    initializer=float(param_init),
                                    trainable=self.agent.learn_penalty,
                                    dtype=tf.float32)
            # penalty = tf.exp(penalty_param)
            self.penalty = tf.nn.softplus(self.penalty_param)

        if self.agent.learn_penalty:
            if self.agent.penalty_param_loss:
                penalty_loss = -self.penalty_param \
                               * (self.cur_cost_ph - self.cost_lim)
            else:
                penalty_loss = -self.penalty \
                               * (self.cur_cost_ph - self.cost_lim)
            self.train_penalty = \
                MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)

    def create_policy_learning_graph(self):
        # Likelihood ratio
        ratio = tf.exp(self.logp - self.logp_old_ph)

        # Surrogate advantage / clipped surrogate advantage
        if self.agent.clipped_adv:
            min_adv = tf.where(self.adv_ph > 0,
                               (1 + self.agent.clip_ratio) * self.adv_ph,
                               (1 - self.agent.clip_ratio) * self.adv_ph
                               )
            surr_adv = tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        else:
            surr_adv = tf.reduce_mean(ratio * self.adv_ph)

        # Surrogate cost
        self.surr_cost = tf.reduce_mean(ratio * self.cadv_ph)

        # Create policy objective function, including entropy regularization
        pi_objective = surr_adv + self.ent_reg * self.ent

        # Possibly include surr_cost in pi_objective
        if self.agent.objective_penalized:
            pi_objective -= self.penalty * self.surr_cost
            pi_objective /= (1 + self.penalty)

        # Loss function for pi is negative of pi_objective
        self.pi_loss = -pi_objective

        # Optimizer-specific symbols
        if self.agent.trust_region:

            # Symbols needed for CG solver for any trust region method
            pi_params = get_vars('pi')
            flat_g = tro.flat_grad(self.pi_loss, pi_params)
            v_ph, hvp = tro.hessian_vector_product(self.d_kl, pi_params)
            if self.agent.damping_coeff > 0:
                hvp += self.agent.damping_coeff * v_ph

            # Symbols needed for CG solver for CPO only
            flat_b = tro.flat_grad(self.surr_cost, pi_params)

            # Symbols for getting and setting params
            get_pi_params = tro.flat_concat(pi_params)
            set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)

            training_package = dict(flat_g=flat_g,
                                    flat_b=flat_b,
                                    v_ph=v_ph,
                                    hvp=hvp,
                                    get_pi_params=get_pi_params,
                                    set_pi_params=set_pi_params)

        elif self.agent.first_order:

            # Optimizer for first-order policy optimization
            train_pi = \
                MpiAdamOptimizer(learning_rate=self.agent.pi_lr).minimize(self.pi_loss)

            # Prepare training package for agent
            training_package = dict(train_pi=train_pi)

        else:
            raise NotImplementedError

        # Provide training package to agent
        training_package.update(dict(pi_loss=self.pi_loss,
                                     surr_cost=self.surr_cost,
                                     d_kl=self.d_kl,
                                     target_kl=self.target_kl,
                                     cost_lim=self.cost_lim))
        self.agent.prepare_update(training_package)

    def create_value_learning_graph(self, vf_lr):
        # Value losses
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v) ** 2)
        self.vc_loss = tf.reduce_mean((self.cret_ph - self.vc) ** 2)

        # If agent uses penalty directly in reward function, don't train a separate
        # value function for predicting cost returns. (Only use one vf for r - p*c.)
        if self.agent.reward_penalized:
            total_value_loss = self.v_loss
        else:
            total_value_loss = self.v_loss + self.vc_loss

        # Optimizer for value learning
        self.train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)

    def creat_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Sync params across processes
        self.sess.run(sync_all_params())
        # Setup model saving
        self.logger.setup_tf_saver(self.sess,
                                   inputs={'x': self.x_ph},
                                   outputs={'pi': self.pi,
                                            'v': self.v, 'vc': self.vc})
        self.agent.prepare_session(self.sess)

    def update(self):
        cur_cost = self.logger.get_stats('EpCost')[0]
        c = cur_cost - self.cost_lim
        if c > 0 and self.agent.cares_about_cost:
            self.logger.log('Warning! Safety constraint is already violated.',
                            'red')
        # =====================================================================#
        #  Prepare feed dict                                                  #
        # =====================================================================#

        inputs = {k: v for k, v in zip(self.buf_phs, self.buf.get())}
        inputs[self.surr_cost_rescale_ph] = self.logger.get_stats('EpLen')[0]
        inputs[self.cur_cost_ph] = cur_cost

        # =====================================================================#
        #  Make some measurements before updating                             #
        # =====================================================================#

        measures = dict(LossPi=self.pi_loss,
                        SurrCost=self.surr_cost,
                        LossV=self.v_loss,
                        Entropy=self.ent)
        if not (self.agent.reward_penalized):
            measures['LossVC'] = self.vc_loss
        if self.agent.use_penalty:
            measures['Penalty'] = self.penalty

        pre_update_measures = self.sess.run(measures, feed_dict=inputs)
        self.logger.store(**pre_update_measures)

        # =====================================================================#
        #  Update penalty if learning penalty                                 #
        # =====================================================================#
        if self.agent.learn_penalty:
            self.sess.run(self.train_penalty,
                          feed_dict={self.cur_cost_ph: cur_cost})

        # =====================================================================#
        #  Update policy                                                      #
        # =====================================================================#
        self.agent.update_pi(inputs)

        # =====================================================================#
        #  Update value function                                              #
        # =====================================================================#
        for _ in range(self.vf_iters):
            self.sess.run(self.train_vf, feed_dict=inputs)

        # =====================================================================#
        #  Make some measurements after updating                              #
        # =====================================================================#

        del measures['Entropy']
        measures['KL'] = self.d_kl

        post_update_measures = self.sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta' + k] = post_update_measures[k] \
                                      - pre_update_measures[k]
        self.logger.store(KL=post_update_measures['KL'], **deltas)

    def env_interact(self, env, epochs, max_ep_len,
                     actual_epoch, save_freq=1, test=False):
        cur_penalty = 0
        cum_cost = 0
        goal_met = False
        o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
        for epoch in range(actual_epoch, epochs + actual_epoch):
            met_goal_step = 'unknown'
            if self.agent.use_penalty:
                cur_penalty = self.sess.run(self.penalty)

            for t in range(self.local_steps_per_epoch):
                # Possibly render
                if self.render and proc_id() == 0 \
                        and t < 1000 and test == False:
                    env.render()

                # Get outputs from policy
                get_action_outs = \
                    self.sess.run(self.get_action_ops,
                                feed_dict={self.x_ph: o[np.newaxis]})
                a = get_action_outs['pi']
                v_t = get_action_outs['v']
                # Agent may not use cost value func
                vc_t = get_action_outs.get('vc', 0)
                logp_t = get_action_outs['logp_pi']
                pi_info_t = get_action_outs['pi_info']
                # Step in environment
                o2, r, d, info = env.step(a)

                # Include penalty on cost
                c = info.get('cost', 0)

                # Track cumulative cost over training
                cum_cost += c

                # save and log
                if self.agent.reward_penalized:
                    r_total = r - cur_penalty * c
                    r_total = r_total / (1 + cur_penalty)
                    self.buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t)
                else:
                    self.buf.store(o, a, r, v_t, c, vc_t, logp_t, pi_info_t)
                self.logger.store(VVals=v_t, CostVVals=vc_t)

                o = o2
                ep_ret += r
                ep_cost += c
                ep_len += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal or (t == self.local_steps_per_epoch - 1):

                    # If trajectory didn't reach terminal state, bootstrap value target(s)
                    if d and not (ep_len == max_ep_len):
                        # Note: we do not count env time out as true terminal state
                        last_val, last_cval = 0, 0
                    else:
                        feed_dict = {self.x_ph: o[np.newaxis]}
                        if self.agent.reward_penalized:
                            last_val = self.sess.run(self.v, feed_dict=feed_dict)
                            last_cval = 0
                        else:
                            last_val, last_cval = \
                                self.sess.run([self.v, self.vc],
                                              feed_dict=feed_dict)
                    self.buf.finish_path(last_val, last_cval)

                    # Only save EpRet / EpLen if trajectory finished
                    if terminal:
                        self.logger.store(EpRet=ep_ret,
                                          EpLen=ep_len, EpCost=ep_cost)
                    else:
                        print('Warning: trajectory cut off '
                              'by epoch at %d steps.' % ep_len)


                    if test: # for test env, reset until reach the ep_len
                        o, r, d, c, ep_ret, ep_len, ep_cost = \
                            env.reset(), 0, False, 0, 0, 0, 0
                    else: # for formal env, do not reset the env and start next episode at current state
                        r, d, c, ep_ret, ep_len, ep_cost = \
                            0, False, 0, 0, 0, 0
                    if env.goal_met() and test == False:  # if goal met, record bool and step
                        met_goal_step = (epoch) * self.steps_per_epoch + t + 1
                        goal_met = True
                        env.done = False
                        env.reset()

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                print('save', epoch)
                self.logger.save_state({'env': env}, None)

            # =====================================================================#
            #  Run RL update                                                      #
            # =====================================================================#
            self.update()

            # =====================================================================#
            #  Cumulative cost calculations                                       #
            # =====================================================================#
            cumulative_cost = mpi_sum(cum_cost)
            cost_rate = cumulative_cost / ((epoch + 1) * self.steps_per_epoch)

            # =====================================================================#
            #  Log performance and stats                                          #
            # =====================================================================#

            self.logger.log_tabular('Epoch', epoch)

            # Performance stats
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpCost', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('CumulativeCost', cumulative_cost)
            self.logger.log_tabular('CostRate', cost_rate)
            self.logger.log_tabular('MetGoalStep', met_goal_step)

            self.logger.log_tabular('DistGoal', env.dist_goal())

            # Surr cost and change
            self.logger.log_tabular('SurrCost', average_only=True)
            self.logger.log_tabular('DeltaSurrCost', average_only=True)

            # V loss and change
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)

            # Anything from the agent?
            self.agent.log()

            # Policy stats
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)

            # Time and steps elapsed
            self.logger.log_tabular('TotalEnvInteracts',
                                    (epoch) * self.steps_per_epoch + t + 1)
            self.logger.log_tabular('Time', time.time() - self.start_time)

            # Show results!
            self.logger.dump_tabular()
            if goal_met:  # if goal met, record bool and step
                break


    def generate_test_env(self, epoch):
        hazard_pos = [0, 1]
        test_config = self.config.copy()
        test_config['hazards_num'] = 1
        del test_config['robot_locations']
        test_config['hazards_locations'] = \
            [coord_to_safety_gym_pos(hazard_pos, self.world_shape)]
        test_config['goal_locations'] = \
            [coord_to_safety_gym_pos([0, 0], self.world_shape)]

        # sample agent initial position
        np.random.seed()
        i = np.random.uniform(-1, 1)
        j = np.random.uniform(-1, 1)
        # make agent outside the hazards or goal
        i += np.sign(i) * 0.5
        j += np.sign(j) * 0.5

        # iteratively set agent near hazards and goal
        if epoch % 2 == 0:
            desire_pos = coord_to_safety_gym_pos([hazard_pos[0] + i,
                                                  hazard_pos[1] + j],
                                                 self.world_shape,
                                                 (1, 1))
        else:
            desire_pos = coord_to_safety_gym_pos([i, j],
                                                 self.world_shape,
                                                 (1, 1))

        test_config['robot_locations'] = [desire_pos]
        env = Engine(test_config)
        print('create')

        return env

    # Experiment loop
    def main_loop(self, test_num, epochs, max_ep_len, save_freq=1):
        self.start_time = time.time()
        for test_epoch in range(test_num):
            test_env = self.generate_test_env(test_epoch)
            self.env_interact(test_env, 1, max_ep_len,
                              test_epoch, save_freq, test=True)
        self.env_interact(self.env, epochs, max_ep_len, test_num,save_freq)
