"""
training a policy from randomly initialized weights
 * Copyright (C) 2021 {Jaekyung Cho} <{jackyoung96@snu.ac.kr}>
 * 
 * This file is part of MAML-RL implementation project, ARIL, SNU
 * 
 * permission of Jaekyung Cho
 
"""

from rl2.agents.ppo import PPOAgent
from PPO.agent import Metalearner, Sampler

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import yaml
import numpy as np

import envs
import gym
import random
import cv2

from tqdm import trange

def random_init(args):
    
    # environment define
    with open(args.config ,'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    env = gym.make(config['env-name'], **config['env-kwargs'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
  
    # maml variables
    fast_batches = config['fast_batches']


    # initialize a PPO agent
    ppo_agent = Metalearner(state_dim, action_dim, config)

    # evaluation phase
    print("Baseline2 evaluation")

    max_eval_step = config['max_eval_step']
    evaluation_num = config['evaluation_num']
    rewards_total = np.zeros(max_eval_step + 1)
    env.seed(100)
    tasks = env.eval_sample_tasks(evaluation_num)

    for eval,task in enumerate(tasks):
        
        del ppo_agent
        ppo_agent = Metalearner(state_dim, action_dim, config) # ppo initialize

        for step in range(config['max_eval_step'] + 1):
            if not step == 0:
                for k in range(fast_batches):
                    state = env.reset(task=task)
                    done = False
                    for t in range(100):
                        action = ppo_agent.select_action(state)
                        state, reward, done, info = env.step(action)

                        ppo_agent.buffer.rewards.append(reward)
                        ppo_agent.buffer.is_terminals.append(done)

                        # if done:
                        #     break

                ppo_agent.update() # single gradient update

            # visualize and evaluation
            state = env.reset(task=task)
            done = False
            reward_sum = 0
            for t in range(100):
                action = ppo_agent.select_action(state)
                state, reward, done, info = env.step(action)
                reward_sum += reward
                if eval == 0:
                    env.render()
                if done:
                    break
            if eval == 0:
                img = env.render(mode='rgb_array', text="baseline2 %d update"%step)
                cv2.imwrite('results/baseline2_%d_update.png'%step, img)
            ppo_agent.buffer.clear() # buffer clear
            # print("%d-th step avg reward %.2f"%(step, reward_sum/(t+1)))
            rewards_total[step] += reward_sum
        
    rewards_total /= evaluation_num

    with open("results/baseline2_result.txt", 'w') as f:
        f.write("gradient_step\treward\n")
        for i,r in enumerate(rewards_total.tolist()):
            f.write("%d\t%.4f\n"%(i,r))

    env.close()
            
 


if __name__ =='__main__':
    import argparse
    
    parser = argparse.ArgumentParser("Baseline2 Training")
    parser.add_argument('--config', type=str, default="configs/2d-navigation.yaml")
    # parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--render', action="store_true")

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--use-cuda', action='store_true')
    misc.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.device = ('cuda:%d'%args.gpu if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    random_init(args)