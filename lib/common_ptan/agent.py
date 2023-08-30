"""
Agent is something which converts states into actions and has state
"""

import copy

import numpy as np
import torch
import random
import torch.nn.functional as F
from . import actions

from torch.autograd import Variable


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model.
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)  

class MO_DDQN_HER():
    """
    Multi-objective DDQN agent 
    """

    def __init__(self, net, action_selector, device, args, preprocessor=default_states_preprocessor):
        super().__init__()
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device
        self.tgt_net = copy.deepcopy(net)
        self.net = net
        self.args = args
        if  self.args.cuda:
            self.net.cuda()
            self.tgt_net.cuda()
        self.state_size = args.obs_shape
        self.action_size =  args.action_shape
        self.reward_size =  args.reward_size
        self.preference = None
        self.w_ep = None
        self.weight_num = args.weight_num
        self.episode_preferences = []
        self.w_batch = []
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.interp = []
        self.tau = args.tau
        self.deterministic = False
        self.total_it = 0

    def __call__(self, states, preference, deterministic = False):
        # Set type for states
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        if preference is None:
            preference = self.preference
        # Randomly select preference from the preference space
        if preference is None:
            if self.w_ep is None:
                self.w_ep = (torch.tensor(self.w_batch[random.choice(np.arange(0,len(self.w_batch))),:])).type(torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor)
                self.w_ep = torch.round(self.w_ep*(10**3))/(10**3)

            preference = self.w_ep   
        with torch.no_grad():
            # Obtain Q values, forward pass DDQN
            Q = self.net(Variable(states.unsqueeze(0)),Variable(preference.unsqueeze(0)))
            Q = Q.view(-1, self.net.reward_size)
            Q = torch.mv(Q.data, preference)
        if not deterministic:
            actions = self.action_selector(Q)
        else:
            actions = np.argmax(Q.cpu().numpy(), axis=0)
        
        return actions
    
    # Learn from batch
    def learn(self, batch, writer):
        
        FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.args.cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if self.args.cuda else torch.ByteTensor
        
        # Unpack the batch 
        def unpack_batch(batch):
            states, actions, rewards, terminals, next_states, preferences = [],[],[],[],[],[]
            for _, exp in enumerate(batch):
                state = np.array(exp.state)
                states.append(state)
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)
                terminals.append(exp.terminal)
                preferences.append(exp.preference)
            return np.array(states, copy=False), np.array(actions), \
                   np.array(rewards, dtype=np.float32), \
                   np.array(terminals, dtype=np.uint8), \
                   np.array(next_states, copy=False),\
                   np.array(preferences, dtype=np.float32, copy=False)
        # Generate non_terminal indices from the batch
        def find_non_terminal_idx(terminal_batch):
            mask = ByteTensor(terminal_batch)
            inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
            inds = inds[mask.eq(0)]
            return inds
        
        
        
        # Unpack the batch 
        states, actions, rewards, terminals, next_states, preferences = unpack_batch(batch)
        state_batch = torch.tensor(states).to(self.device)
        action_batch = torch.tensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards).to(self.device)
        next_state_batch = torch.tensor(next_states).to(self.device)
        terminal_batch = torch.tensor(terminals).to(self.device)
        w_batch = torch.tensor(preferences).to(self.device)
        # Project preferences to normalized value space
        w_batch_sim = torch.from_numpy(self.interp(w_batch)).type(FloatTensor).to(self.device)
        Q = self.net(state_batch, w_batch)
        # Extend preferences to match the dimensions of Q           
        w_ext = w_batch.unsqueeze(2).repeat(1, self.net.action_size, 1)
        w_ext = w_ext.view(-1, self.net.reward_size)
        # Extend projected preferences to match the dimensions of Q 
        w_ext_sim = w_batch_sim.unsqueeze(2).repeat(1, self.net.action_size, 1)
        w_ext_sim = w_ext_sim.view(-1, self.net.reward_size)
        # Find non-terminal transitions
        mask_non_terminal = find_non_terminal_idx(terminal_batch)
        
        with torch.no_grad():
            # Transform negative penalty to positive penalty to calculate cosine similarity for DST
            if self.args.scenario_name == 'dst-v0':
                target_Q = self.tgt_net(next_state_batch,w_batch)
                tmp_target_Q = target_Q.view(-1, self.net.reward_size)
                tmp_target_Q_sim = torch.zeros(tmp_target_Q.shape).to(self.device)
                tmp_target_Q_sim[:,1] = 19
                tmp_target_Q_sim = tmp_target_Q_sim + tmp_target_Q
                # Obtain actions that maximizes preference-driven optimality term
                act = (torch.clamp(F.cosine_similarity(w_ext_sim,tmp_target_Q_sim),0, 0.9999)*torch.bmm(w_ext.unsqueeze(1),tmp_target_Q.unsqueeze(2)).squeeze()).view(-1, self.net.action_size).max(1)[1]
            else:
                target_Q = self.tgt_net(next_state_batch,w_batch)
                tmp_target_Q = target_Q.view(-1, self.net.reward_size)
                # Obtain actions that maximizes preference-driven optimality term
                act = (torch.clamp(F.cosine_similarity(w_ext_sim,tmp_target_Q),0, 0.9999)*torch.bmm(w_ext.unsqueeze(1),tmp_target_Q.unsqueeze(2)).squeeze()).view(-1, self.net.action_size).max(1)[1]
            target_Q = target_Q.gather(1, act.view(-1, 1, 1).expand(target_Q.size(0), 1, target_Q.size(2))).squeeze()
             
            Tau_Q = torch.zeros(self.args.batch_size, self.net.reward_size).type(FloatTensor)
            Tau_Q[mask_non_terminal] = self.args.gamma * target_Q[mask_non_terminal]
            Tau_Q += reward_batch
    
        actions_tmp = action_batch
        actions_tmp = actions_tmp.type(torch.int64)
        # Obtain Q-values
        Q_org = Q.gather(1, actions_tmp.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))).view(-1, self.net.reward_size)

    
        
        # Compute Loss
        loss = F.smooth_l1_loss(Q_org, Tau_Q)
        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for param, target_param in zip(self.net.parameters(), self.tgt_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
       
        self.total_it += 1
        # Write the results to tensorboard
        writer.add_scalar('Loss/Loss', loss, self.total_it)
    
    def reset_preference(self):
        self.w_ep = None


class MO_TD3_HER:

    def __init__(self, actor, critic, device, args, preprocessor=float32_preprocessor):
        
        super().__init__()

        self.args = args
        self.preprocessor = preprocessor
        self.device = device

        if self.args.cuda == True:
            self.actor = actor.cuda()
            if critic:
                self.critic = critic.cuda()
        else:
            self.actor = actor
            self.critic = critic

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)        

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        if critic:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        self.state_size = args.obs_shape
        self.action_size =  args.action_shape
        self.reward_size =  args.reward_size
        self.preference = None
        self.total_it = 0
        self.args = args
        self.w_ep = None
        self.weight_num = args.weight_num
        self.max_action = args.max_action
        self.gamma = args.gamma
        self.tau  = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.expl_noise =args.expl_noise
        self.policy_freq = args.policy_freq
        self.deterministic = False
        self.w_batch = []
        self.interp = []

    def __call__(self, states, preference, deterministic = False):

        # Set type for states
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        if preference is None:
            preference = self.preference
        # Randomly select preference from the preference space
        if preference is None:
            if self.w_ep is None:
                self.w_ep = (torch.tensor(self.w_batch[random.choice(np.arange(0,len(self.w_batch))),:])).type(torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor)
                self.w_ep = torch.round(self.w_ep*(10**3))/(10**3)
                
            preference = self.w_ep
       
        # Choose action for a given policy 
        if not deterministic:
            actions = self.actor(states.unsqueeze(0), preference.unsqueeze(0)).cpu().numpy().flatten()
            # TD3 Exploration
            actions = (actions + np.random.normal(0,self.max_action*self.expl_noise,size=self.action_size)).clip(-self.max_action,self.max_action)
        else:
            actions = self.actor(states.unsqueeze(0), preference.unsqueeze(0)).cpu().data.numpy().flatten()
        return actions
    
    # Learn from batch
    def learn(self, batch, writer):
        
        FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.args.cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if self.args.cuda else torch.ByteTensor
        
        self.total_it += 1
        
        # Unpack the batch 
        def unpack_batch(batch):
            states, actions, rewards, terminals, next_states, preferences = [],[],[],[],[],[]
            for _, exp in enumerate(batch):
                states.append(np.array(exp.state))
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)
                terminals.append(exp.terminal)
                preferences.append(exp.preference)
            return np.array(states, copy=False,dtype=np.float32), np.array(actions,dtype=np.float32), \
                   np.array(rewards, dtype=np.float32), \
                   np.array(terminals, dtype=np.uint8), \
                   np.array(next_states, copy=False,dtype=np.float32),\
                   np.array(preferences, dtype=np.float32, copy=False)

        # Generate non_terminal indices from the batch
        def find_non_terminal_idx(terminal_batch):
            mask = ByteTensor(terminal_batch)
            inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
            inds = inds[mask.eq(0)]
            return inds
        
        
        
        # Unpack the batch 
        states, actions, rewards, terminals, next_states, preferences = unpack_batch(batch)
        state_batch = torch.tensor(states).to(self.device)
        action_batch = torch.tensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards).to(self.device)
        next_state_batch = torch.tensor(next_states).to(self.device)
        terminal_batch = torch.tensor(terminals).to(self.device)
        w_batch = torch.tensor(preferences).to(self.device)

        w_batch_np_critic = copy.deepcopy(w_batch.cpu().numpy())
        w_batch_np_actor = copy.deepcopy(w_batch.cpu().numpy())

        # Find non-terminal transitions
        mask_non_terminal = find_non_terminal_idx(terminal_batch)

 
        with torch.no_grad():
                       
            # Compute the target Q value
            noise = (torch.randn_like(torch.tensor(actions))*self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            next_action_batch= torch.FloatTensor((self.actor_target(next_state_batch, w_batch) + noise).cpu().numpy().clip(-self.max_action, self.max_action)).to(self.device)
            # Project preferences to normalized value space
            w_batch_np_critic = self.interp(w_batch_np_critic)
            w_batch_critic_loss = torch.from_numpy(w_batch_np_critic).type(FloatTensor).to(self.device) 
            
            target_Q1, target_Q2 = self.critic_target(next_state_batch, w_batch, next_action_batch)

           
            wTauQ1 = torch.bmm(w_batch.unsqueeze(1),target_Q1.unsqueeze(2)).squeeze()
            wTauQ2 = torch.bmm(w_batch.unsqueeze(1),target_Q2.unsqueeze(2)).squeeze()
            _, wTauQ_min_idx = torch.min(torch.cat((wTauQ1.unsqueeze(-1),wTauQ2.unsqueeze(-1)),dim=-1),1)
            
            Tau_Q = torch.zeros(self.args.batch_size ,self.args.reward_size).type(FloatTensor).to(self.device)
            idx = 0
            for ind in wTauQ_min_idx:
                if ind == 0:
                    Tau_Q[idx,:] = target_Q1[idx,:]
                else:
                    Tau_Q[idx,:] = target_Q2[idx,:]
                idx += 1    
            
            target_Q = torch.zeros(self.args.batch_size,self.args.reward_size).type(FloatTensor).to(self.device)
            target_Q[mask_non_terminal] = self.gamma * Tau_Q[mask_non_terminal]
            target_Q += reward_batch
                     
            
        
        # Get current Q values
        current_Q1, current_Q2 = self.critic(state_batch, w_batch, action_batch)


        angle_term_1 = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_critic_loss,current_Q1),0, 0.9999)))
        angle_term_2 = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_critic_loss,current_Q2),0, 0.9999)))

        # Compute Critic Loss
        critic_loss = angle_term_1.mean() +  F.smooth_l1_loss(current_Q1, target_Q) + \
                    angle_term_2.mean() +  F.smooth_l1_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm= 100)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute Actor Loss
            Q = self.critic.Q1(state_batch, w_batch, self.actor(state_batch,w_batch))            
            wQ = torch.bmm(w_batch.unsqueeze(1),Q.unsqueeze(2)).squeeze()
            actor_loss = -wQ  

            # Project preferences to normalized value space
            w_batch_np_actor = self.interp(w_batch_np_actor)                     
            w_batch_actor_loss = torch.from_numpy(w_batch_np_actor).type(FloatTensor).to(self.device)
            angle_term = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_actor_loss,Q),0, 0.9999)))
            
           
            actor_loss = actor_loss.mean() + self.args.actor_loss_coeff*angle_term.mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm= 100)
            
            self.actor_optimizer.step()
                       
            
            # Soft update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
            
                         
            # Write the results to tensorboard
            if (self.total_it % 1000) == 0:
                writer.add_scalar('Loss/Actor_Loss'.format(), actor_loss, self.total_it)
                writer.add_scalar('Loss/Critic_Loss'.format(), critic_loss, self.total_it)
        
    
    def reset_preference(self):
        self.w_ep = None           

class MO_TD3_HER_Key:

    def __init__(self, actor, critic, device, args, preprocessor=float32_preprocessor):
        
        super().__init__()

        self.args = args
        self.preprocessor = preprocessor
        self.device = device

        if self.args.cuda == True:
            self.actor = actor.cuda()
            if critic:
                self.critic = critic.cuda()
        else:
            self.actor = actor
            self.critic = critic

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)        

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        if critic:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        self.state_size = args.obs_shape
        self.action_size =  args.action_shape
        self.reward_size =  args.reward_size
        self.preference = None
        self.total_it = 0
        self.args = args
        self.w_ep = None
        self.max_action = args.max_action
        self.gamma = args.gamma
        self.tau  = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.expl_noise =args.expl_noise
        self.deterministic = False


    def __call__(self, states, preference, deterministic = False):

        FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        # Set type for states
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        
        # Choose action according to policy for a given preference  
        if not deterministic:
            pref_noise = (np.random.normal(size = preference.shape)*0.05).clip(-0.05, 0.05)
            preference = FloatTensor(preference+pref_noise)
            preference = (torch.abs(preference)/torch.norm(preference, p=1))
            self.w_ep = preference
            actions = self.actor(states.unsqueeze(0), preference.unsqueeze(0)).cpu().numpy().flatten()
            actions = (actions + np.random.normal(0,self.max_action*self.expl_noise,size=self.action_size)).clip(-self.max_action,self.max_action)
        else:
            preference = FloatTensor(preference)
            self.w_ep = preference
            actions = self.actor(states.unsqueeze(0), preference.unsqueeze(0)).cpu().data.numpy().flatten()
        return actions
    
    # Learn from batch
    def learn(self, batch, writer):
        
        FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.args.cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if self.args.cuda else torch.ByteTensor
        
        self.total_it += 1
        
        # Unpack the batch 
        def unpack_batch(batch):
            states, actions, rewards, terminals, next_states, preferences = [],[],[],[],[],[]
            for _, exp in enumerate(batch):
                states.append(np.array(exp.state))
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)
                terminals.append(exp.terminal)
                preferences.append(exp.preference)
            return np.array(states, copy=False,dtype=np.float32), np.array(actions,dtype=np.float32), \
                   np.array(rewards, dtype=np.float32), \
                   np.array(terminals, dtype=np.uint8), \
                   np.array(next_states, copy=False,dtype=np.float32),\
                   np.array(preferences, dtype=np.float32, copy=False)
        # Generate non_terminal indices from the batch
        def find_non_terminal_idx(terminal_batch):
            mask = ByteTensor(terminal_batch)
            inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
            inds = inds[mask.eq(0)]
            return inds
        
        
        
        # Unpack the batch 
        states, actions, rewards, terminals, next_states, preferences = unpack_batch(batch)

        state_batch = torch.tensor(states).to(self.device)
        action_batch = torch.tensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards).to(self.device)
        next_state_batch = torch.tensor(next_states).to(self.device)
        terminal_batch = torch.tensor(terminals).to(self.device)
        w_batch = torch.tensor(preferences).to(self.device)

        # Find non-terminal transitions
        mask_non_terminal = find_non_terminal_idx(terminal_batch)

        
        with torch.no_grad():
                       
            # Compute the target Q value
            noise = (torch.randn_like(torch.tensor(actions))*self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            next_action_batch= torch.FloatTensor((self.actor_target(next_state_batch, w_batch) + noise).cpu().numpy().clip(-self.max_action, self.max_action)).to(self.device)
            target_Q1, target_Q2 = self.critic_target(next_state_batch, w_batch, next_action_batch)
            wTauQ1 = torch.bmm(w_batch.unsqueeze(1),target_Q1.unsqueeze(2)).squeeze()
            wTauQ2 = torch.bmm(w_batch.unsqueeze(1),target_Q2.unsqueeze(2)).squeeze()
            _, wTauQ_min_idx = torch.min(torch.cat((wTauQ1.unsqueeze(-1),wTauQ2.unsqueeze(-1)),dim=-1),1)
            
            Tau_Q = torch.zeros(self.args.batch_size ,self.args.reward_size).type(FloatTensor).to(self.device)
            idx = 0
            for ind in wTauQ_min_idx:
                if ind == 0:
                    Tau_Q[idx,:] = target_Q1[idx,:]
                else:
                    Tau_Q[idx,:] = target_Q2[idx,:]
                idx += 1    

            
            target_Q = torch.zeros(self.args.batch_size,self.args.reward_size).type(FloatTensor).to(self.device)
            target_Q[mask_non_terminal] = self.gamma * Tau_Q[mask_non_terminal]
            target_Q += reward_batch
                   
            
        
        # Get current Q values
        current_Q1, current_Q2 = self.critic(state_batch, w_batch, action_batch)
                
       # Compute Critic Loss
        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) +  F.smooth_l1_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm= 100)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute Actor Loss
            Q = self.critic.Q1(state_batch, w_batch, self.actor(state_batch,w_batch))
            wQ = torch.bmm(w_batch.unsqueeze(1),Q.unsqueeze(2)).squeeze()         
            actor_loss = -wQ  
            actor_loss = actor_loss.mean()  
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm= 100)
            self.actor_optimizer.step()
                       
            
            # Soft update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
             
            # Write the results to tensorboard
            if writer:
                writer.add_scalar('Loss/Actor_Loss'.format(), actor_loss, self.total_it)
                writer.add_scalar('Loss/Critic_Loss'.format(), critic_loss, self.total_it)
        