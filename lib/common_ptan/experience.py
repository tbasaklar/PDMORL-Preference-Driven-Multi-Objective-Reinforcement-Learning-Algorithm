import gym
import torch
import numpy as np

from collections import namedtuple, deque


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state','terminal', 'preference', 'step_idx', 'p_id', 'info'])      


class MORLExperienceSource:
    """
    Experience source MORL
    """
    def __init__(self, env, agent, args, steps_count=1, steps_delta=1):

        assert isinstance(steps_count, int)
        assert steps_count >= 1
        self.env = env
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.args = args
        self.iter_idx = 0
        self.multi_objective_key = False
        self.multi_objective_key_preference = []
        if hasattr(args, 'p_id'):
            self.p_id = args.p_id
        else:
            self.p_id = 0


    def __iter__(self):
        history, cur_rewards = [], 0
        self.iter_idx = 0
        global_steps = 0
        s = self.env.reset() 
        if self.args.reward_size > 1:
            cur_rewards = np.zeros((self.args.reward_size)) 
            cur_steps = 0
        else:
            cur_rewards = 0
            cur_steps = 0

        
        while True:   
            u = []
            with torch.no_grad():
                if self.multi_objective_key:
                    action = self.agent(s, preference = self.multi_objective_key_preference)
                else:
                    action = self.agent(s, preference = None)
                if hasattr(self.args,'start_timesteps'):
                    if global_steps < self.args.start_timesteps:
                            action  = self.env.action_space.sample()
                u = action
            s_next, r, done, info = self.env.step(u)
            cur_rewards += r
            history.append(Experience(state=s, action=u, reward=r, next_state = s_next, terminal = done, preference = self.agent.w_ep.cpu().numpy(), step_idx=self.iter_idx, p_id=self.p_id, info = []))
            cur_steps += 1
            global_steps += 1
            if len(history) >= self.steps_count:
                yield tuple(history)
            
                history.clear()
            
            s = s_next
            self.iter_idx += 1
            if done:
                self.iter_idx = 0
                if hasattr(self.agent, 'reset_preference'):
                    self.agent.reset_preference() 
                s = self.env.reset()
                if self.args.reward_size > 1:
                    self.total_rewards.append(cur_rewards)
                    cur_rewards = np.zeros((self.args.reward_size)) 
                    self.total_steps.append(cur_steps)
                    cur_steps = 0
                else:
                    self.total_rewards.append(cur_rewards)
                    self.total_steps.append(cur_steps)
                    cur_rewards = 0
                    cur_steps = 0
            
            
    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
        return r
    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res
  
class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size, random_sample = True):
        """
        Get one random batch from experience replay
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        if random_sample:
            return [self.buffer[key] for key in keys]
        else:
            return self.buffer

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample[0])
        else:
            for _i in range(len(sample)): 
                self.buffer.pop(0)
            self.buffer.append(sample[0])

    def populate(self, samples):
        """
        Populates samples into the buffer
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)                
            self._add(entry)

    def clear(self):
        self.buffer = []
       

class ExperienceReplayBuffer_HER_MO(ExperienceReplayBuffer):
    
    """
    Multi-objective Hindsight Experience Replay Buffer 
    """
    def __init__(self, experience_source, args):
        assert isinstance(experience_source, (MORLExperienceSource,type(None)))
        super().__init__(experience_source, args.replay_size)
        self.args = args
        self.experience_source = experience_source
        self.time_steps = args.time_steps
        self.ep_p = np.zeros((args.process_count))
        self.w_all_count = 0
    
    
    def populate(self, samples):
        """
        Populates samples into the buffer
        """
        self._add(samples) 
    
    
    
    def sample(self, batch_size):

        inds = np.random.choice(len(self.buffer), batch_size, replace=True)
        batch = [self.buffer[i] for i in inds]

        return batch 
    
    def _add(self, sample):

        #Generate N_w preferences for HER
        w_batch_rnd = np.random.randn(self.args.weight_num, self.args.reward_size)
        w_batch = np.abs(w_batch_rnd) / np.linalg.norm(w_batch_rnd, ord=1, axis=1, keepdims=True)
        w_batch = np.round(w_batch,3)
        
        samples = sample*(self.args.weight_num+1)
        if len(self.buffer) < self.capacity:
            self.buffer.append(samples[-1])
            if len(self.buffer) > self.args.start_timesteps*self.args.process_count:
                for i in range(self.args.weight_num):
                    sample_prime = samples[i]
                    sample_prime = sample_prime._replace(preference = w_batch[i])
                    self.buffer.append(sample_prime)

        else:
           
            for _i in range(self.args.weight_num+1): 
                self.buffer.pop(0)
            self.buffer.append(samples[-1])
            for i in range(self.args.weight_num):
                sample_prime = samples[i]
                sample_prime = sample_prime._replace(preference = w_batch[i])
                self.buffer.append(sample_prime)

    
