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


Queue_obj = namedtuple('Queue_obj', ['ep_samples', 'time_step','ep_cnt','process_ID'])

def child_process(actor, critic, args, train_queue,p_id,w_batch):
        torch.manual_seed(p_id*args.seed)
        random.seed(p_id*args.seed)
        np.random.seed(p_id*args.seed)
        os.environ['PYTHONHASHSEED'] = str(p_id*args.seed)
        torch.backends.cudnn.deterministic = True
        env = gym.make(args.scenario_name)
        env.seed(p_id*args.seed)
        args.p_id = p_id

        # Initialize agent and source
        agent = ptan.agent.MO_TD3_HER(actor,critic, args.device, args)
        exp_source =  ptan.experience.MORLExperienceSource(env, agent, args, steps_count=1)
        done_episodes = 0
        time_step = 0
        agent.w_batch = w_batch
        # Interact with the environment
        for time_step, exp in enumerate(exp_source):
            #Check if the terminal condition is reached
            if exp[0].terminal:
                done_episodes +=1
            train_queue.put(Queue_obj(ep_samples = copy.deepcopy(exp), time_step = time_step, ep_cnt = done_episodes, process_ID = p_id))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    start_time = time.time()
    name = "HalfCheetah_MO_TD3_HER"
    args = lib.utilities.settings.HYPERPARAMS[name]
    args.plot_name  = name
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
    args.action_shape = env_main.action_space.shape[0]
    args.reward_size = len(env_main.reward_space)
    args.max_action = env_main.action_space.high
    args.max_episode_len = env_main._max_episode_steps
          
    
    #Writer object for Tensorboard
    log_dir = "runs/" + datetime.now().strftime("%d %b %Y, %H-%M") + ' MO_TD3_HER '+args.scenario_name
    writer = SummaryWriter(log_dir, comment = ' TD3_HER'+args.scenario_name)
    #Initialize the networks
    actor = lib.models.networks.Actor(args)
    critic = lib.models.networks.Critic(args)

    #Edit the neural network model name
    args.name_model = args.name

    #Load previously trained model
    if args.load_model == True:
        load_path = "Exps/{}/".format(name)
        model_actor = torch.load("{}{}.pkl".format(load_path,"{}_{}_{}".format(args.scenario_name, args.name_model,'final_actor'))) # Change the model name accordingly
        actor = lib.models.networks.Actor(args)
        actor.load_state_dict(model_actor)
        model_critic = torch.load("{}{}.pkl".format(load_path,"{}_{}_{}".format(args.scenario_name, args.name_model,'final_critic'))) # Change the model name accordingly
        critic = lib.models.networks.Critic(args)
        critic.load_state_dict(model_critic)
        
    actor.share_memory()
    critic.share_memory()


    #Initialize preference spaces
    w_batch_test = lib.utilities.MORL_utils.generate_w_batch_test(args, step_size = args.w_step_size)
    w_batch_eval = lib.utilities.MORL_utils.generate_w_batch_test(args, step_size = 0.005)
    w_batch_test_split = np.array_split(w_batch_test,PROCESSES_COUNT)
    # Initialize child processes
    train_queue_list = []
    data_proc_list = []
    for p_id in range(PROCESSES_COUNT):
        train_queue = mp.JoinableQueue(maxsize=1)
        data_proc = mp.Process(target=child_process,
                               args=(actor, critic, args, train_queue,p_id,w_batch_test_split[p_id]))
        data_proc.start()
        train_queue_list.append(train_queue)
        data_proc_list.append(data_proc)

    # Join Queue objects
    for p_id in range(PROCESSES_COUNT):
        train_queue_list[p_id].join()
 
    #Initialize RL agent
    agent_main = ptan.agent.MO_TD3_HER(actor,critic, device, args)
    
    #Initialize Experience Source and Replay Buffer
    exp_source_main = ptan.experience.MORLExperienceSource(env_main, agent_main, args, steps_count=1)
    replay_buffer_main = ptan.experience.ExperienceReplayBuffer_HER_MO(exp_source_main, args)

    # Initialize Interpolator
    x = np.loadtxt("interp_objs_halfCheetah.txt",delimiter=",")
    x_unit = x/np.linalg.norm(x,ord=2,axis=1,keepdims=True)
    idx_w_batch = np.round(np.linspace(0, len(w_batch_test)-1, num=len(x))).astype(int)
    w_batch_interp = w_batch_test[idx_w_batch]
    interp = RBFInterpolator(w_batch_interp, x_unit, kernel= 'linear')
    agent_main.interp = interp

    # Main Loop
    done_episodes = 0
    time_step = 0
    process_step_array = np.zeros((PROCESSES_COUNT,))
    process_episode_array = np.zeros((PROCESSES_COUNT,))
    eval_cnt = 1
    eval_cnt_ep = 1
    try:
        for ts in tqdm(range(0, PROCESSES_COUNT*args.time_steps,PROCESSES_COUNT)): #iterate through the fixed number of timesteps
            
            for main_cnt in range(PROCESSES_COUNT):
                process_samples = train_queue_list[main_cnt].get()
                process_step_array[process_samples.process_ID] = process_samples.time_step
                process_episode_array[process_samples.process_ID] = process_samples.ep_cnt
                replay_buffer_main.populate(process_samples.ep_samples)

            if len(replay_buffer_main.buffer) < 2*args.batch_size*args.weight_num:
                # Reading from Queue is done
                for main_cnt in range(PROCESSES_COUNT):
                    train_queue_list[main_cnt].task_done()
                continue 

            # Learn from the minibatch
            for i in range(PROCESSES_COUNT):
                batch = replay_buffer_main.sample(args.batch_size) 
                agent_main.learn(batch, writer)

            for main_cnt in range(PROCESSES_COUNT):
                train_queue_list[main_cnt].task_done()
                       
            # Update Interpolator
            if (process_episode_array > (eval_cnt_ep)).all():
                eval_cnt_ep +=1
                x_tmp = lib.utilities.MORL_utils.eval_agent_interp(test_env_main, agent_main, w_batch_interp, args, eval_episodes=args.eval_episodes)
                for obj_num in range(len(x)):
                    scalarized_obj_prev = np.dot(w_batch_interp[obj_num],x[obj_num])
                    scalarized_obj_current = np.dot(w_batch_interp[obj_num],x_tmp[obj_num])
                    if scalarized_obj_current > scalarized_obj_prev:
                        print(f"Previous Objective: {x[obj_num]}, Current Objective: {x_tmp[obj_num]}, Preference: {w_batch_interp[obj_num]}")
                        x[obj_num] = x_tmp[obj_num]
                x_unit = x/np.linalg.norm(x,ord=1,axis=1,keepdims=True)        
                interp = RBFInterpolator(w_batch_interp, x_unit, kernel= 'linear')
                agent_main.interp = interp
            time_step = int(sum(process_step_array))
            # Evaluate agent
            if (process_episode_array > (args.eval_freq*eval_cnt)).all():
                eval_cnt +=1
                hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_agent(test_env_main, agent_main, w_batch_eval, args, eval_episodes=args.eval_episodes)
                print(f"Time steps of Each Process: {process_step_array}, Episode Count of Each Process: {process_episode_array}")
                #Store episode results and write to tensorboard
                lib.utilities.MORL_utils.store_results( [], hypervolume, sparsity, time_step, writer, args)
                lib.utilities.common_utils.save_model(actor, args, name = name, ext ='actor_{}'.format(time_step))
                lib.utilities.common_utils.save_model(critic, args, name = name,ext ='critic_{}'.format(time_step))
                lib.utilities.MORL_utils.plot_objs(args,objs,ext='{}'.format(time_step))
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()      
    print(f"Total Number of Time Steps: {sum(process_step_array)}, Total Number of Episodes: {sum(process_episode_array)}")
    #Evaluate the final agent
    hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_agent(test_env_main, agent_main, w_batch_test, args, eval_episodes=args.eval_episodes)
    lib.utilities.MORL_utils.store_results([], hypervolume, sparsity, time_step, writer, args)
    lib.utilities.common_utils.save_model(actor, args, name = name,ext ='final_actor')
    lib.utilities.common_utils.save_model(critic, args, name = name,ext ='final_critic')
    lib.utilities.MORL_utils.plot_objs(args,objs,ext='final')
    print("Done in %d steps and %d episodes!" % (time_step, sum(process_episode_array)))
    print("Time Consumed")
    print("%0.2f minutes" % ((time.time() - start_time)/60))
    writer.close()
    

    
