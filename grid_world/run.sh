#
# Author: Akifumi Wachi
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

for i in `seq 100`
do
mkdir results/$i
mkdir myenv/params/$i

ENV_PARAM_FILE=myenv/params/$i/env_settings.npz

# Oracle agent
python main.py sim_type=oracle idx_sim=$i env.env_param_file=$ENV_PARAM_FILE env.reuse_env=False

# Random agent
python main.py sim_type=random idx_sim=$i env.env_param_file=$ENV_PARAM_FILE env.reuse_env=True

# Proposed method
python main.py sim_type=safe_glm idx_sim=$i env.env_param_file=$ENV_PARAM_FILE env.reuse_env=True

# Safe GLM Stepwise agent
python main.py sim_type=safe_glm_stepwise idx_sim=$i env.env_param_file=$ENV_PARAM_FILE env.reuse_env=True

# Unsafe agent
python main.py sim_type=unsafe_glm idx_sim=$i env.env_param_file=$ENV_PARAM_FILE env.reuse_env=True

# GP-based agent (context) agent
python main.py sim_type=safe_gp_feature idx_sim=$i env.env_param_file=$ENV_PARAM_FILE env.reuse_env=True

# GP-based agent (state) agent
python main.py sim_type=safe_gp_state idx_sim=$i env.env_param_file=$ENV_PARAM_FILE env.reuse_env=True
done
