from __future__ import absolute_import, division, print_function
import sys
# importing time module
import time

from tensorboardX import SummaryWriter
from datetime import datetime
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
from scipy.interpolate import RBFInterpolator

from collections import namedtuple, deque
import copy

time_obj = [-1, -3, -5, -7, -8, -9, -13, -14, -17, -19]
treasure_obj = [0.7, 8.2, 11.5, 14., 15.1, 16.1, 19.6, 20.3, 22.4, 23.7]

Queue_obj = namedtuple('Queue_obj', ['ep_samples', 'time_step','ep_cnt','process_ID'])


def child_process(net, args, train_queue,p_id,w_batch):
        torch.manual_seed(p_id*args.seed)
        random.seed(p_id*args.seed)
        np.random.seed(p_id*args.seed)
        os.environ['PYTHONHASHSEED'] = str(p_id*args.seed)
        torch.backends.cudnn.deterministic = True
        env = gym.make(args.scenario_name)
        env.seed(p_id*args.seed)
        args.p_id = p_id

        #Initialize action selection policy
        selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=args.epsilon_start) 
        #Initialize epsilon tracker if the action selection policy requires epsilon
        if args.epsilon_decay:
            epsilon_tracker = lib.utilities.common_utils.EpsilonTracker(selector, args)
        # Initialize agent and source
        agent = ptan.agent.MO_DDQN_HER(net, selector, args.device, args)
        exp_source =  ptan.experience.MORLExperienceSource(env, agent, args, steps_count=1)
        done_episodes = 0
        time_step = 0
        agent.w_batch = w_batch
        # Interact with the environment
        for time_step, exp in enumerate(exp_source):
            #Check if the terminal condition is reached
            if exp[0].terminal:
                done_episodes +=1
                agent.episode_preferences.clear()
                epsilon_tracker.update(time_step)
            train_queue.put(Queue_obj(ep_samples = copy.deepcopy(exp), time_step = time_step, ep_cnt = done_episodes, process_ID = p_id))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    start_time = time.time()
    args = lib.utilities.settings.HYPERPARAMS["DeepSeaTreasure_MO_DDQN_HER"]
    PROCESSES_COUNT = args.process_count
    torch.set_num_threads(PROCESSES_COUNT)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    # setup the environment
    env_main = gym.make(args.scenario_name)
    env_main.seed(args.seed)
    test_env_main = gym.make(args.scenario_name)
    test_env_main.seed(args.seed)
    
    #Initialize environment related arguments
    args.obs_shape = env_main.observation_space.shape[0]
    args.action_shape = env_main.action_space.n
    
    args.reward_size = len(env_main.reward_space)
         
    
    #Writer object for Tensorboard
    log_dir = "runs/" + datetime.now().strftime("%d %b %Y, %H-%M") + ' MO_DDQN_HER '+args.scenario_name
    writer = SummaryWriter(log_dir, comment = ' MO_DDQN_HER'+args.scenario_name)
    #Initialize the network
    net = lib.models.networks.MO_DDQN(args)
    net.share_memory()

    #Initialize preference spaces
    w_batch_test = lib.utilities.MORL_utils.generate_w_batch_test(args,step_size = args.w_step_size)
    w_batch_test_split = np.array_split(w_batch_test,PROCESSES_COUNT)
    train_queue_list = []
    # Initialize child processes
    data_proc_list = []
    for p_id in range(PROCESSES_COUNT):
        train_queue = mp.JoinableQueue(maxsize=1)
        data_proc = mp.Process(target=child_process,
                               args=(net, args, train_queue,p_id,w_batch_test_split[p_id]))
        data_proc.start()
        train_queue_list.append(train_queue)
        data_proc_list.append(data_proc)

    #Initialize action selection policy
    selector_main = ptan.actions.EpsilonGreedyActionSelector(epsilon=args.epsilon_start) 
    #Initialize epsilon tracker if the action selection policy requires epsilon
    if args.epsilon_decay:
        epsilon_tracker_main = lib.utilities.common_utils.EpsilonTracker(selector_main, args)

    #Edit the neural network model name
    args.name_model = args.name
    
   
    #Initialize RL agent
    agent_main = ptan.agent.MO_DDQN_HER(net, selector_main, device, args)
    
    #Initialize Experience Source and Replay Buffer
    exp_source_main = ptan.experience.MORLExperienceSource(env_main, agent_main, args, steps_count=1)
    replay_buffer_main = ptan.experience.ExperienceReplayBuffer_HER_MO(exp_source_main, args)
    # Initialize Interpolator
    # apply discount
    dst_time = (-(1 - np.power(args.gamma, -np.asarray(time_obj))) / (1 - args.gamma)) + 19
    dst_treasure = (np.power(args.gamma, -np.asarray(time_obj) - 1) * np.asarray(treasure_obj))    
    x = np.vstack((dst_treasure,dst_time)).transpose()
    w_batch_interp = np.zeros((args.reward_size+1,args.reward_size))
    for i in range(args.reward_size):
        tmp_array = np.zeros((args.reward_size,))
        tmp_array[i] = 1
        w_batch_interp[i,:] = tmp_array
    w_batch_interp[-1,:] = np.ones((args.reward_size,))*(1/args.reward_size)
    x_new = np.zeros((args.reward_size+1,args.reward_size))
    for i in range(args.reward_size+1):
        x_tmp = []
        for ii in range(len(x)):
            x_tmp.append(np.dot(w_batch_interp[i], x[ii])) 
        x_new[i]=x[np.array(x_tmp).argmax()]

    x_unit = x_new/np.linalg.norm(x_new,ord=2,axis=1,keepdims=True)
    
    
    interp = RBFInterpolator(w_batch_interp, x_unit, kernel= 'linear')
    agent_main.interp = interp


    # Main Loop
    total_rewards = []
    process_step_array = np.zeros((PROCESSES_COUNT,))
    process_episode_array = np.zeros((PROCESSES_COUNT,))
    update_priority_buffer_flag = 0
    max_metric = 0
    eval_cnt = 1
    try:
        for ts in tqdm(range(0, PROCESSES_COUNT*args.time_steps,PROCESSES_COUNT)): #iterate through the fixed number of timesteps
            for main_cnt in range(PROCESSES_COUNT):
                process_samples = train_queue_list[main_cnt].get()
                process_step_array[process_samples.process_ID] = process_samples.time_step
                process_episode_array[process_samples.process_ID] = process_samples.ep_cnt
                replay_buffer_main.populate(process_samples.ep_samples)
            if len(replay_buffer_main.buffer) < 2*args.batch_size*args.weight_num:
                continue 
            for i in range(PROCESSES_COUNT):
                batch = replay_buffer_main.sample(args.batch_size) 
                # Learn from the minibatch
                agent_main.learn(batch, writer)
            
            time_step = int(sum(process_step_array))
            # Interpolator is not updated since we know the Pareto front set of solutions for discrete problems beforehand
            # Evaluate agent
            if sum(process_episode_array) > (PROCESSES_COUNT*args.eval_freq*eval_cnt):
                eval_cnt +=1
                CR_F1, hypervolume, sparsity, _ = lib.utilities.MORL_utils.eval_agent_discrete(test_env_main, agent_main, w_batch_test, args)
                print(f"Time steps of Each Process: {process_step_array}, Episode Count of Each Process: {process_episode_array}")
                #Store episode results and write to tensorboard
                lib.utilities.MORL_utils.store_results(CR_F1, hypervolume, sparsity,time_step, writer, args)
                model_eval_metric  = CR_F1
                # Store best model
                if max_metric <= model_eval_metric:
                    max_metric = model_eval_metric
                    max_metric_time_step = time_step
                    lib.utilities.common_utils.save_model(net, args, name = 'DeepSeaTreasure_MO_DDQN_HER', ext ='{}'.format(time_step))
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()      

    #Evaluate the final agent
    CR_F1, hypervolume, sparsity, _ = lib.utilities.MORL_utils.eval_agent_discrete(test_env_main, agent_main, w_batch_test, args)
    #Store episode results and write to tensorboard
    lib.utilities.MORL_utils.store_results(CR_F1, hypervolume, sparsity,time_step, writer, args)
    model_eval_metric  = CR_F1
    # Store best model
    if max_metric <= model_eval_metric:
        max_metric = model_eval_metric
        max_metric_time_step = time_step
        lib.utilities.common_utils.save_model(net, args, name = 'DeepSeaTreasure_MO_DDQN_HER',ext ='{}'.format(max_metric_time_step))
    print("Done in %d steps and %d episodes!" % (time_step, sum(process_episode_array)))
    print("Time Consumed")
    print("%0.2f minutes" % ((time.time() - start_time)/60))
    writer.close()
    

    
