# PD-MORL

This codebase contains the implementation for the paper [PD-MORL: Preference-Driven Multi-Objective Reinforcement Learning Algorithm](<https://openreview.net/forum?id=zS9sRyaPFlJ>) (**ICLR 2023**). 

In this paper, we propose a novel preference-driven multi-objective reinforcement learning algorithm using a single policy network that covers the entire preference space in a given domain.

![teaser](images/teaser.gif)

## Installation

#### Prerequisites

- **Operating System**: tested on Ubuntu 18.04.
- **Python Version**: >= 3.8.11.
- **PyTorch Version**: >= 1.8.1.
- **MuJoCo** : install mujoco and mujoco-py of version 2.1 by following the instructions in [mujoco-py](<https://github.com/openai/mujoco-py>).

#### Dependencies

You can either install the dependencies in a conda virtual env or manually. 

For conda virtual env installation, simply create a virtual env named **pdmorl** by:

```
conda env create -f environment.yml
```

If you prefer to do it manually, you could just simply open the `environment.yml` in the editor to install packages using `pip`.

You still need to satisfy the Prerequisites.
For example, to install pytorch 1.8.1 with the cuda 11.1 toolkit, simply run the following command 
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Repo Details

All training and evaluation-related codes are in the folder `PD-MORL`. We also provide some common utilities and environments in the `lib` folder. We follow a similar structure to [PTAN](<https://github.com/Shmuma/ptan>) in our codebase for readibility and simplicity.

#### MORL Benchmarks
We evaluate PD-MORL's performance on two commonly used discrete MORL benchmarks: Deep Sea Treasure and Fruit Tree Navigation. For these benchmarks, we use the implementation of [https://github.com/RunzheYang/MORL](<https://github.com/RunzheYang/MORL>). We also evaluate its performance on multi-objective continuous control tasks such as MO-Walker2d-v2, MO-HalfCheetah-v2, MO-Ant-v2, MO-Swimmer-v2, and MO-Hopper-v2. For these benchmarks, we use the implementation of [https://github.com/mit-gfx/PGMORL](<https://github.com/mit-gfx/PGMORL>).

The code for the environments can be found in `lib/utilities/morl/moenvs` folder. You should register these environments as a gym environment by:

```
cd ../lib/utilities/morl/MOEnvs
pip install -e .
```

Additionally, if needed, you may comment out the following line 21 in this gym script `home/[user_id]/.conda/envs/[env_name]/lib/python[version]/site-packages/gym/wrappers/time_limit.py`:
```
# info["TimeLimit.truncated"] = not done
```
#### Pretrained Models and Evaluation

You can find the pretrained models in `EvaluationNetworks` folder. Using these models, you can run 

```
python eval_benchmarks_MO_TD3_HER.py --benchmark_name 'walker' --model_ext 'actor' --eval_episodes 1
```

for continuous controls tasks. In this code, change the benchmark name and the model name accordingly.
You can run either of these commands for discrete benchmarks

```
python test_DeepSeaTreasure_MO_DDQN_HER.py --model_ext 950370
python test_FruitTreeNavigation_MO_DDQN_HER.py --tree_depth 6 --model_ext 312050
```
In this code, change the model name accordingly.

You can also find precomputed Pareto front solutions  in `PD-MORL` as `objs_pdmorl_[problem name]`.


#### Training

We provide a configuration of hyperparameters for each benchmark in `lib/utilities/settings.py`. You can modify these hyperparameters according to your needs.
We provide a training script in the `PD-MORL` folder for each benchmark. For instance, for the MO-Ant-v2 problem, you can run the following command to start training:

```
cd PD-MORL
python train_Walker2d_MO_TD3_HER.py
```

By default, the models are stored in `Exps/[problem_algorithm name]` during training. The Pareto front plots are stored in `Figures/[problem name]` for continuous benchmarks.


#### Computational Resource

We run all our experiments on a local server, including Intel Xeon Gold 6242R @4.1GHz and 385 GB memory. We do not use any GPU in our implementation. 

## Citation

If you find our paper or code is useful, please consider citing: 

```
@inproceedings{basaklar2022pd,
  title={PD-MORL: Preference-Driven Multi-Objective Reinforcement Learning Algorithm},
  author={Basaklar, Toygun and Gumussoy, Suat and Ogras, Umit},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```
