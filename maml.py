"""
MAML
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

def change_distribution(array):
    return np.sign(array)*0.5 + array

def maml(args):
    
    # environment define
    with open(args.config ,'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    env = gym.make(config['env-name'], **config['env-kwargs'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    num_batches = config['num_batches']   
    update_timestep = config['update_timestep']     
    save_path = config['save_path_maml']

    # maml variables
    fast_batches = config['fast_batches']
    meta_batches = config['meta_batches']


    # initialize a PPO agent
    ppo_agent = Metalearner(state_dim, action_dim, config)
    if args.pretrained:
        ppo_agent.load(save_path)

    sampler = Sampler(state_dim, action_dim, config)


    # training
    env.seed(1)
    if not args.test:
        # outer loop
        for episode in trange(1, num_batches + 1):
            tasks = env.sample_tasks(fast_batches)

            for task in tasks: # sample batch of tasks

                state = env.reset(task=task)

                # initialize a sampler
                sampler.policy.actor.load_state_dict(ppo_agent.policy.actor.state_dict())
                sampler.buffer.clear()

                for k in range(1): # sample K trajectories (just 1 trajectory)

                    state = env.reset(task=task)
                    done = False  

                    for t in range(update_timestep):  # maximum time step (H)
                        action, _ = sampler.select_action(state)
                        state,reward,done,info = env.step(action)

                        # if t == update_timestep - 1:
                        #     done = True

                        if args.render:
                            env.render()

                        # state, action, logprob는 select_action에서 buffer에 들어간다
                        sampler.buffer.rewards.append(reward)
                        sampler.buffer.is_terminals.append(done)

                        # if done:
                        #     break

                # sampler update
                sampler.update()
                sampler.buffer.clear()

                # trajectory sampling for meta learning
                for k in range(meta_batches):

                    state = env.reset(task=task)
                    done = False

                    for t in range(update_timestep):  # maximum time step (H)
                        action, logprob = sampler.select_action(state) # trajectory by sampler
                        state,reward,done,info = env.step(action)

                        # if t == update_timestep - 1:
                        #     done = True

                        if args.render:
                            env.render()

                        # ppo buffer
                        ppo_agent.buffer.states.append(torch.Tensor(state))
                        ppo_agent.buffer.actions.append(torch.Tensor(action))
                        ppo_agent.buffer.logprobs.append(torch.Tensor(logprob))
                        ppo_agent.buffer.rewards.append(reward)
                        ppo_agent.buffer.is_terminals.append(done)

                        # if done:
                        #     break
                
                sampler.buffer.clear()

            # meta update
            ppo_agent.update()

            ppo_agent.save(save_path)

    # evaluation phase
    print("MAML evaluation")

    max_eval_step = config['max_eval_step']
    evaluation_num = config['evaluation_num']
    rewards_total = np.zeros(max_eval_step + 1)
    env.seed(100)
    tasks = env.eval_sample_tasks(evaluation_num)

    for eval,task in enumerate(tasks):
        ppo_agent = Metalearner(state_dim, action_dim, config)
        ppo_agent.load(save_path) # ppo initialize to meta-trained status

        for step in range(max_eval_step + 1):
            
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
                    img = env.render(mode='rgb_array', text="maml %d update"%step)
                if done:
                    break
            if eval == 0:
                img = env.render(mode='rgb_array', text="maml %d update"%step)
                cv2.imwrite('results/maml_%d_update.png'%step, img)
            ppo_agent.buffer.clear() # buffer clear
            # print("%d-th step avg reward %.2f"%(step, reward_sum/(t+1)))
            rewards_total[step] += reward_sum

        del ppo_agent
        
    rewards_total /= evaluation_num

    with open("results/maml_result.txt", 'w') as f:
        f.write("gradient_step\treward\n")
        for i,r in enumerate(rewards_total.tolist()):
            f.write("%d\t%.4f\n"%(i,r))

    env.close()
            
 


if __name__ =='__main__':
    import argparse
    
    parser = argparse.ArgumentParser("MAML Training")
    parser.add_argument('--config', type=str, default="configs/2d-navigation.yaml")
    # parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--render', action="store_true")

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--use-cuda', action='store_true')
    misc.add_argument('--pretrained', action='store_true')
    misc.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.device = ('cuda:%d'%args.gpu if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    maml(args)