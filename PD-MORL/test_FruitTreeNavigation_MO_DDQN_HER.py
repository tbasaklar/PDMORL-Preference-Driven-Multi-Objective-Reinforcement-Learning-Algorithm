from __future__ import absolute_import, division, print_function
import sys
import time

import torch

sys.path.append('../')
import lib

import lib.common_ptan as ptan

import numpy as np
import gym
import moenvs
import argparse


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_depth', type=int, default= 6)
    parser.add_argument('--model_ext', type=str, default='')
    term_args = parser.parse_args()

    start_time = time.time()
    args = lib.utilities.settings.HYPERPARAMS["FruitTreeNavigation_MO_DDQN_HER"]
    PROCESSES_COUNT = args.process_count
    torch.set_num_threads(PROCESSES_COUNT)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.depth = term_args.tree_depth
    # setup the environment
    env = gym.make(args.scenario_name, depth= args.depth)
    test_env = gym.make(args.scenario_name, depth= args.depth)
    
    #Initialize environment related arguments
    args.obs_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.n
    
    args.reward_size = len(env.reward_space)        
    
    #Initialize action selection policy
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=args.epsilon_start) 
    #Initialize epsilon tracker if the action selection policy requires epsilon
    if args.epsilon_decay:
        epsilon_tracker = lib.utilities.common_utils.EpsilonTracker(selector, args)
        
    
    #Edit the neural network model name
    args.name_model = args.name
    
    w_batch_test = lib.utilities.MORL_utils.generate_w_batch_test(args,step_size = 0.1)
    
    print('############ MO-DDQN-HER ############')
    agent = None
    #Initialize the network
    net_test = lib.models.networks.MO_DDQN(args)
    #Initialize RL agent
    agent = ptan.agent.MO_DDQN_HER(net_test, selector, device, args)
    load_path = "EvaluationNetworks/{}/".format(args.scenario_name)
    # Load Trained Model 
    ################################### Change the model number accordingly ###################################
    model = torch.load("{}{}.pkl".format(load_path,"{}_{}_{}_{}".format(args.scenario_name, args.name_model,term_args.model_ext,args.depth))) 
    net_test.load_state_dict(model)
    CR_F1, hypervolume, sparsity, _ = lib.utilities.MORL_utils.eval_agent_discrete(test_env, agent, w_batch_test, args)
    print("CRF1 %0.2f; Hypervolume %0.2f; Sparsity %0.2f" % (CR_F1, hypervolume, sparsity))
    
        
    print('############ Envelope Approach ############')
    args.gamma = 0.99
    net_test = lib.models.networks.EnvelopeLinearCQN_default(args)
    #Initialize RL agent
    agent = ptan.agent.MO_DDQN_HER(net_test, selector, device, args)
    load_path = "EvaluationNetworks/{}/".format(args.scenario_name)
    # Load Trained Model
    model = torch.load("{}ftn_default{}.pkl".format(load_path,args.depth),map_location='cpu')
    net_test.load_state_dict(model)
    CR_F1, hypervolume, sparsity, _ = lib.utilities.MORL_utils.eval_agent_discrete(test_env, agent, w_batch_test, args)
    print("CRF1 %0.2f; Hypervolume %0.2f; Sparsity %0.2f" % (CR_F1, hypervolume, sparsity))

    

    
