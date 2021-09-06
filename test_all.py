"""
Test MAML
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

def main(args):
    with open(args.config ,'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    env = gym.make(config['env-name'], **config['env-kwargs'])

    with open(args.config ,'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    env = gym.make(config['env-name'], **config['env-kwargs'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    
    save_path_maml = config['save_path_maml']
    save_path_bs1 = config['save_path_baseline1']
    max_eval_step = config['max_eval_step']

    # maml variables
    fast_batches = config['fast_batches']


    ## maml test
    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.seed(1000)
    # different task distribution
    tasks = [{'goal': np.random.rand(2)-np.array([1.5,1.5])} for _ in range(100)] # 하던 중
    maml_agent = Metalearner(state_dim, action_dim, config)
    maml_agent.load(save_path_maml) # ppo initialize to meta-trained status

    rewards_total = np.zeros(max_eval_step)
    for task in tasks:
        for step in range(max_eval_step):

            for k in range(fast_batches):
                state = env.reset(task=task)
                done = False
                for t in range(100):
                    action = maml_agent.select_action(state)
                    state, reward, done, info = env.step(action)

                    maml_agent.buffer.rewards.append(reward)
                    maml_agent.buffer.is_terminals.append(done)

            maml_agent.update() # single gradient update

            # visualize and evaluation
            state = env.reset(task=task)
            done = False
            reward_sum = 0
            for t in range(100):
                action = maml_agent.select_action(state)
                state, reward, done, info = env.step(action)
                reward_sum += reward
                # img = env.render(mode='rgb_array', text="maml %d update"%step)
                if done:
                    break
            # cv2.imwrite('results/maml_%d_update.png'%step, img)
            maml_agent.buffer.clear() # buffer clear
            # print("%d-th step avg reward %.2f"%(step, reward_sum/(t+1)))
            rewards_total[step] += reward_sum / 100

    with open("results/maml_result.txt", 'w') as f:
        f.write("gradient_step\treward\n")
        for i,r in enumerate(rewards_total.tolist()):
            f.write("%d\t%.4f\n"%(i,r))

    del maml_agent
    env.close()
    
    
    ## baseline 1
    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.seed(1000)
    bs1_agent = Metalearner(state_dim, action_dim, config)
    bs1_agent.load(save_path_bs1) # ppo initialize to pretrained status 

    rewards_total = np.zeros(max_eval_step)
    for task in tasks:
        for step in range(config['max_eval_step']):

            for k in range(fast_batches):
                state = env.reset(task=task)
                done = False
                for t in range(100):
                    action = bs1_agent.select_action(state)
                    state, reward, done, info = env.step(action)

                    bs1_agent.buffer.rewards.append(reward)
                    bs1_agent.buffer.is_terminals.append(done)

            bs1_agent.update() # single gradient update

            # visualize and evaluation
            state = env.reset(task=task)
            done = False
            reward_sum = 0
            for t in range(100):
                action = bs1_agent.select_action(state)
                state, reward, done, info = env.step(action)
                reward_sum += reward
                # img = env.render(mode='rgb_array', text="baseline1 %d update"%step)
                if done:
                    break
            # cv2.imwrite('results/baseline1_%d_update.png'%step, img)
            bs1_agent.buffer.clear() # buffer clear
            # print("%d-th step avg reward %.2f"%(step, reward_sum/(t+1)))
            rewards_total[step] += reward_sum / 100
    
    with open("results/baseline1_result.txt", 'w') as f:
        f.write("gradient_step\treward\n")
        for i,r in enumerate(rewards_total.tolist()):
            f.write("%d\t%.4f\n"%(i,r))
    
    del bs1_agent
    env.close()

    # baseline 2
    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.seed(1000)
    bs2_agent = Metalearner(state_dim, action_dim, config) # ppo initialize

    rewards_total = np.zeros(max_eval_step)
    for task in tasks:
        for step in range(config['max_eval_step']):

            for k in range(fast_batches):
                state = env.reset(task=task)
                done = False
                for t in range(100):
                    action = bs2_agent.select_action(state)
                    state, reward, done, info = env.step(action)

                    bs2_agent.buffer.rewards.append(reward)
                    bs2_agent.buffer.is_terminals.append(done)

            bs2_agent.update() # single gradient update

            # visualize and evaluation
            state = env.reset(task=task)
            done = False
            reward_sum = 0
            for t in range(100):
                action = bs2_agent.select_action(state)
                state, reward, done, info = env.step(action)
                reward_sum += reward
                # img = env.render(mode='rgb_array', text="baseline2 %d update"%step)
                if done:
                    break
            # cv2.imwrite('results/baseline2_%d_update.png'%step, img)
            bs2_agent.buffer.clear() # buffer clear
            # print("%d-th step avg reward %.2f"%(step, reward_sum/(t+1)))
            rewards_total[step] += reward_sum / 100
    
    with open("results/baseline2_result.txt", 'w') as f:
        f.write("gradient_step\treward\n")
        for i,r in enumerate(rewards_total.tolist()):
            f.write("%d\t%.4f\n"%(i,r))

    del bs2_agent
    env.close()



    env.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("MAML")
    parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=3)

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    args.device = ('cuda:%d'%args.gpu if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
    