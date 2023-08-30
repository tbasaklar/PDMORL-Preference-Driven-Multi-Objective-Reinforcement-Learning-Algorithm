from __future__ import absolute_import, division, print_function
import sys
import time


import torch
import random
import torch.optim as optim
import torch.multiprocessing as mp
import os

sys.path.append('../')
from tqdm import tqdm
import lib

import lib.common_ptan as ptan

import numpy as np

import gym
import moenvs
import itertools
from collections import namedtuple

def generate_w_batch_test(step_size, reward_size):
    mesh_array = []
    step_size = step_size
    for i in range(reward_size):
        mesh_array.append(np.arange(0,1+step_size, step_size))
        
    w_batch_test = np.array(list(itertools.product(*mesh_array)))
    w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1,:]
    w_batch_test = np.unique(w_batch_test,axis =0)
    
    return w_batch_test

# Evaluate agent for X episodes and returns average reward
def eval_agent(test_env, agent, args, preference, eval_episodes=1):
    
    use_cuda = args.cuda
    # use_cuda = False
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    avg_reward = np.zeros((eval_episodes,))
    avg_multi_obj = np.zeros((eval_episodes,args.reward_size))
    for eval_ep in range(eval_episodes):
        test_env.seed(eval_ep*10)
        test_env.action_space.seed(eval_ep*10)
        torch.manual_seed(eval_ep*10)
        np.random.seed(eval_ep*10)    
        # reset the environment
        state = test_env.reset()
        terminal = False
        tot_rewards = 0
        multi_obj = 0
        cnt = 0
        # Interact with the environment
        while not terminal:
            if hasattr(agent, 'deterministic'):
                action = agent(state, preference, deterministic = True)
            else:
                action = agent(state, preference)
            next_state, reward, terminal, info = test_env.step(action)
            tot_rewards += np.dot(preference,reward)
            multi_obj +=  reward#
            state = next_state
            cnt += 1
        avg_reward[eval_ep]= tot_rewards
        avg_multi_obj[eval_ep]= multi_obj
    return avg_reward, avg_multi_obj

Queue_obj = namedtuple('Queue_obj', ['avg_reward', 'avg_multi_obj','process_ID'])
def child_process(preference, p_id,train_queue):

        start_time = time.time()
        args = lib.utilities.settings.HYPERPARAMS["Walker2d_MO_TD3_HER_Key"]
        args.p_id = p_id
        
        device = torch.device("cuda" if args.cuda else "cpu")
        args.device = device
        # setup the environment
        torch.manual_seed(p_id*args.seed)
        random.seed(p_id*args.seed)
        np.random.seed(p_id*args.seed)
        os.environ['PYTHONHASHSEED'] = str(p_id*args.seed)
        torch.backends.cudnn.deterministic = True
        env = gym.make(args.scenario_name)
        env.seed(p_id*args.seed)
        env = gym.make(args.scenario_name)
        test_env = gym.make(args.scenario_name) 

        #Initialize environment related arguments
        args.obs_shape = env.observation_space.shape[0]
        args.action_shape = env.action_space.shape[0]
        args.reward_size = len(env.reward_space)
        args.max_action = env.action_space.high
        args.max_episode_len = env._max_episode_steps
            
        
        # Initialize critic and actor networks
        actor = lib.models.networks.Actor(args)
        critic = lib.models.networks.Critic(args)
        #Main Objects

        #Edit the neural network model name
        args.name_model = args.name + " -Key"
        
        #Initialize RL agent
        agent = ptan.agent.MO_TD3_HER_Key(actor,critic, device, args)
            
        #Initialize Experience Source and Replay Buffer
        exp_source = ptan.experience.MORLExperienceSource(env, agent, args, steps_count=1)
        # Fix the key preference
        exp_source.multi_objective_key = True
        exp_source.multi_objective_key_preference = preference
        replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size = args.replay_size)
 
        # Main Loop
        done_episodes = 0
        max_metric = 0
        max_multi_obj_reward = 0
        # Only print the progress of first child process
        if p_id == 0:
            disable = False
        else:
            disable  = True
        for ts in tqdm(range(0, args.time_steps), disable=disable): #iterate through the fixed number of timesteps
            
            # Populate transitions
            replay_buffer.populate(1)

            if len(replay_buffer.buffer) < 2*args.batch_size:
                continue 
            
            # Learn from the minibatch 
            batch = replay_buffer.sample(args.batch_size) 
            agent.learn(batch,writer=None)
            
            new_rewards = exp_source.pop_total_rewards()

            if new_rewards:
                done_episodes += 1
                # Evaluate agent
                if done_episodes % args.eval_freq == 0:
                    avg_reward, avg_multi_obj = eval_agent(test_env,agent, args, preference, eval_episodes = 10)
                    avg_reward_mean = avg_reward.mean()
                    avg_reward_std = avg_reward.std()
                    avg_multi_obj_mean = avg_multi_obj.mean(axis=0)
                    avg_multi_obj_std = avg_multi_obj.std(axis=0)
                    if max_metric <= avg_reward_mean:
                        max_metric = avg_reward_mean
                        max_multi_obj_reward = avg_multi_obj_mean
                        queue_obj = Queue_obj(avg_reward = max_metric, avg_multi_obj = max_multi_obj_reward, process_ID = p_id)
                    print("\n---------------------------------------")
                    print(f"Process {args.p_id} - Evaluation Episode {done_episodes}: Reward: {np.round(avg_reward_mean,2)}(std: +-{avg_reward_std}), Multi-Objective Reward: {avg_multi_obj_mean}(std: +-{avg_multi_obj_std}), Max_Metric: {max_multi_obj_reward}, Preference: {preference}")
                    print("---------------------------------------")
        
        
        avg_reward, avg_multi_obj = eval_agent(test_env, agent, args, preference, eval_episodes = 10)
        avg_reward_mean = avg_reward.mean()
        avg_reward_std = avg_reward.std()
        avg_multi_obj_mean = avg_multi_obj.mean(axis=0)
        avg_multi_obj_std = avg_multi_obj.std(axis=0)
        if max_metric <= avg_reward_mean:
            max_metric = avg_reward_mean
            max_multi_obj_reward = avg_multi_obj_mean
            queue_obj = Queue_obj(avg_reward = max_metric, avg_multi_obj = max_multi_obj_reward, process_ID = p_id)
        print("\n---------------------------------------")
        print(f"Process {args.p_id} - Evaluation Episode {done_episodes}: Reward: {np.round(avg_reward_mean,2)}(std: +-{avg_reward_std}), Multi-Objective Reward: {avg_multi_obj_mean}(std: +-{avg_multi_obj_std}), Max_Metric: {max_multi_obj_reward}, Preference: {preference}")
        print("---------------------------------------")
        print("Done in %d steps and %d episodes!" % (ts, done_episodes))
        print("Time Consumed")
        print("%0.2f minutes" % ((time.time() - start_time)/60))
        train_queue.put(queue_obj)

def main_parallel(process_count, reward_size):
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    w_batch = generate_w_batch_test(0.001, reward_size) # w step size and number of objectives
    obj_num = process_count
    idx = np.round(np.linspace(0, len(w_batch)-1, num=obj_num)).astype(int)
    preference_array = w_batch[idx]
    torch.set_num_threads(process_count)
    # Initialize child processes
    train_queue_list = []
    data_proc_list = []
    for id in range(process_count):
        train_queue = mp.Queue(maxsize=1)
        data_proc = mp.Process(target=child_process,
                               args=(preference_array[id],id,train_queue))
        data_proc.start()
        train_queue_list.append(train_queue)
        data_proc_list.append(data_proc)
    for id in range(process_count):
        data_proc_list[id].join()
    return train_queue_list, data_proc_list

if __name__ == "__main__":
    process_count = 3 # Number of key preferences
    reward_size = 2 # 2 objective problem
    train_queue_list, data_proc_list = main_parallel(process_count, reward_size)
    results = np.zeros((process_count,reward_size))
    
    for id in range(process_count):
        result = train_queue_list[id].get()
        results[id] = result[1]

    for p in data_proc_list:
        p.terminate()
        p.join() 

    f = open('interp_objs_walker2d.txt', 'w')
    for t in results:
        line = ','.join(str(x) for x in t)
        f.write(line +' \n')
    f.close()

    
