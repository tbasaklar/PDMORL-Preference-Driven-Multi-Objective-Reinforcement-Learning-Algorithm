from __future__ import absolute_import, division, print_function
import sys
# importing time module
import time


import torch

sys.path.append('../')
import lib

import lib.common_ptan as ptan

import numpy as np
import gym
import moenvs
import argparse

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_name', type=str, default='walker',help='Benchmark name. {walker,halfCheetah,ant,swimmer,hopper2}')
    parser.add_argument('--model_ext', type=str, default='')
    parser.add_argument('--eval_episodes', type=int, default=6)
    term_args = parser.parse_args()
    
    start_time = time.time()
    benchmark_name = term_args.benchmark_name
    if benchmark_name == 'walker':
        name = 'Walker2d_MO_TD3_HER'
    elif benchmark_name == 'halfCheetah':
        name = 'HalfCheetah_MO_TD3_HER'
    elif benchmark_name == 'ant':
        name = 'Ant_MO_TD3_HER'
    elif benchmark_name == 'swimmer':
        name = 'Swimmer_MO_TD3_HER'
    elif benchmark_name == 'hopper2':
        name = 'Hopper_MO_TD3_HER'

    args = lib.utilities.settings.HYPERPARAMS[name]

    torch.set_num_threads(4)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    # setup the environment
    env = gym.make(args.scenario_name)
    test_env = gym.make(args.scenario_name)
    
    #Initialize environment related arguments
    args.obs_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.reward_size = len(env.reward_space)
    args.max_action = env.action_space.high
    args.max_episode_len = env._max_episode_steps
          

    args.name_model = args.name
    #Initialize preference space
    w_batch_test = lib.utilities.MORL_utils.generate_w_batch_test(args,args.w_step_size)
    
    print('############ PD-MORL ############')
    agent = None
    #Initialize the networks
    actor = lib.models.networks.Actor(args)
    critic = lib.models.networks.Critic(args)
    #Initialize RL agent
    agent = ptan.agent.MO_TD3_HER(actor,critic, device, args)
    args.plot_name  = name
    save_path = "EvaluationNetworks/{}/".format(args.scenario_name)
    ################################### Change the model number accordingly ###################################
    # model = torch.load("{}{}.pkl".format(save_path,"{}_{}_{}".format(args.scenario_name, args.name_model,'actor_8206160')))
    model = torch.load("{}{}.pkl".format(save_path,"{}_{}_{}".format(args.scenario_name, args.name_model,term_args.model_ext)))
    actor.load_state_dict(model)
    hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_agent_test(test_env, agent, w_batch_test, args,eval_episodes = term_args.eval_episodes)

    hv_mean = hypervolume.mean()
    hv_std = hypervolume.std()
    sp_mean = sparsity.mean()
    sp_std = sparsity.std()
    objs_mean = objs.mean(axis=0)
    objs_std = objs.std(axis=0)
    print("\n---------------------------------------")
    print(f"Hypervolume: {hv_mean}(std: +-{hv_std}), Sparsity: {sp_mean}(std: +-{sp_std})")
    print("---------------------------------------")
    
    non_dom = NonDominatedSorting().do(-objs_mean, only_non_dominated_front=True)        
    objs_pareto = objs_mean[non_dom]
    
    objs_pareto = np.round(objs_pareto,4)
    np.savetxt('objs_pdmorl_'+benchmark_name+'.txt', objs_pareto, delimiter=',')


    # Uncomment the following line to visualize the Pareto Front
    # lib.utilities.MORL_utils.plot_objs(args,objs_pareto,ext='{}'.format('test'))

    

    

    
