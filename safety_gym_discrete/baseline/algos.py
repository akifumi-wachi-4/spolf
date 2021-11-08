# Author: Akifumi Wachi, Yunyue Wei
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

from safe_rl.pg.agents import (PPOAgent, TRPOAgent, CPOAgent)

default_kwargs = dict(
    reward_penalized=False,
    objective_penalized=False,
    learn_penalty=False,
    penalty_param_loss=False
)

lagrangian_kwargs = dict(
    reward_penalized=False,
    objective_penalized=True,
    learn_penalty=True,
    penalty_param_loss=True
)

def cpo_agent():
    cpo_kwargs = default_kwargs
    agent = CPOAgent(**cpo_kwargs)
    return agent

def ppo_lag_agent():
    ppo_kwargs = lagrangian_kwargs
    agent = PPOAgent(**ppo_kwargs)
    return agent

def trpo_lag_agent():
    trpo_kwargs = lagrangian_kwargs
    agent = TRPOAgent(**trpo_kwargs)
    return agent
