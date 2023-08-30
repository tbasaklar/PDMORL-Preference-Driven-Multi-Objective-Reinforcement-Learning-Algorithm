from math import log
from lib.common_ptan import actions
import sys
import time

import torch
import torch.nn as nn

import copy
from types import SimpleNamespace


import lib.common_ptan as ptan

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 args: SimpleNamespace):
        self.selector = selector
        self.args = copy.deepcopy(args)
        self.update(0)

    def update(self, time_step: int):
        eps = self.args.epsilon_start - time_step / (self.args.time_steps)
        self.selector.epsilon = max(self.args.epsilon_final, eps)
        

def save_model(net, args, name = '', ext=''):
        save_path = "Exps/{}/".format(name)
        model_name = "{}_{}_{}".format(args.scenario_name, args.name_model,ext)
        torch.save(net.state_dict(), "{}{}.pkl".format(save_path, model_name))
