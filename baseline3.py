"""
Oracle
 * Copyright (C) 2021 {Jaekyung Cho} <{jackyoung96@snu.ac.kr}>
 * 
 * This file is part of MAML-RL implementation project, ARIL, SNU
 * 
 * permission of Jaekyung Cho
 
"""

import yaml
import numpy as np

import envs
import gym
import random

from tqdm import trange

class Oracle:
    def __init__(self):
        self._task = None
        self._goal = None

    def select_action(self, state):
        action = self._goal - state
        return action
    
    def reset(self, task):
        self._task = task
        self._goal = task['goal']


def oracle(args):
    
    # environment define
    with open(args.config ,'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    env = gym.make(config['env-name'], **config['env-kwargs'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
  
    # maml variables
    fast_batches = config['fast_batches']
    evaluation_num = config['evaluation_num']

    # define oracle
    oracle = Oracle()

    # evaluation phase
    print("Baseline3 evaluation")
    env.seed(100)
    tasks = env.eval_sample_tasks(evaluation_num)
    for eval,task in enumerate(tasks):

        state = env.reset(task=task)
        oracle.reset(task=task)
        done = False
        reward_sum = 0
        for t in range(100):
            action = oracle.select_action(state)
            state, reward, done, info = env.step(action)
            reward_sum += reward
            if eval==0:
                env.render()
            if done:
                break

    # print("avg reward %.2f"%(reward_sum/(t+1)))
    with open("results/baseline3_result.txt", 'w') as f:
        f.write("gradient_step\treward\n")
        f.write("%d\t%.4f\n"%(0,reward_sum/evaluation_num))

    env.close()
            
 


if __name__ =='__main__':
    import argparse
    
    parser = argparse.ArgumentParser("Baseline3")
    parser.add_argument('--config', type=str, default="configs/2d-navigation.yaml")
    parser.add_argument('--render', action="store_true")

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--test', action='store_true')

    args = parser.parse_args()

    oracle(args)