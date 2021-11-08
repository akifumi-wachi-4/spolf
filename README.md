# Safe Policy Optimization with Local Feature (SPO-LF)
This is the source-code for implementing the algorithms in the paper "Safe Policy Optimization with Local Generalized Linear Function Approximations" which was presented in NeurIPS-21.

## Installation
There is `requirements.txt` in this repository.
Except for the common modules (e.g., numpy, scipy), our source code depends on the following modules.
- Mandatory
  - Gym-MiniGrid (https://github.com/maximecb/gym-minigrid)
  - Hydra (https://github.com/facebookresearch/hydra)
  - pymdptoolbox (https://github.com/sawcordwell/pymdptoolbox)

- Optional
  - GPy (https://github.com/SheffieldML/GPy)

We also provide `Dockerfile` in this repository, which can be used for reproducing our grid-world experiment.

## Simulation configuration
We manage the simulation configuration using [hydra](https://github.com/facebookresearch/hydra). Configurations are listed in `config.yaml`. For example, the algorithm to run should be chosen from the ones we implemented:

```yaml
sim_type: {safe_glm, unsafe_glm, random, oracle, safe_gp_state, safe_gp_feature, safe_glm_stepwise}
```

## Grid World Experiment

The source code necessary for our grid-world experiment is contained in `/grid_world` folder. To run the simulation, for example, use the following commands.

```bash
cd grid_world
python main.py sim_type=safe_glm env.reuse_env=False
```

For the monte carlo simulation while comparing our proposed method with baselines, use the shell file, `run.sh`.

We also provide a script for visualization. If you want to render how the agent behaves, use the following command.

```bash
python main.py sim_type=safe_glm env.reuse_env=True
```

<p align="center">
<img src="/figures/grid_world.gif" width=504 height=376>
</p>

## Safety-Gym Experiment

The source code necessary for our safety-gym experiment is contained in `/safety_gym_discrete` folder. 
Our experiment is based on [safety-gym](https://github.com/openai/safety-gym). Our proposed method utilize dynamic programming algorithms to solve Bellman Equation, so we modified `engine.py` to discrtize the environment. We attach modified safety-gym source code in `/safety_gym_discrete/engine.py`.
To use the modified library, please clone [safety-gym](https://github.com/openai/safety-gym), then replace `safety-gym/safety_gym/envs/engine.py` using `/safety_gym_discrete/engine.py` in our repo. Using the following commands to install the modified library:

```bash
cd safety_gym
pip install -e .
```

Note that [MuJoCo](http://www.mujoco.org/) licence is needed for installing Safety-Gym.
To run the simulation, use the folowing commands.

```bash
cd safety_gym_discrete
python main.py sim_idx=0
```

We compare our proposed method with three notable baselines: CPO, PPO-Lagrangian, and TRPO-Lagrangian. The baseline implementation depends on [safety-starter-agents](https://github.com/openai/safety-starter-agents). We modified `run_agent.py` in the repo source code.

To run the baseline, use the folowing commands.

```bash
cd safety_gym_discrete/baseline
python baseline_run.py sim_type=cpo
```
The environment that agent runs on is generated using `generate_env.py`. We provide 10 50*50 environments. If you want to generate other environments, you can change the world shape in `safety_gym_discrete.py`, and running the following commands:

```bash
cd safety_gym_discrete
python generate_env.py
```

## Citation
If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{wachi_yue_sui_neurips2021,
  Author = {Wachi, Akifumi and Wei, Yunyue and Sui, Yanan},
  Title = {Safe Policy Optimization with Local Generalized Linear Function Approximations},
  Booktitle  = {Neural Information Processing Systems (NeurIPS)},
  Year = {2021}
}
```