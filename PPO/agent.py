import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical



################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__()

        self.has_continuous_action_space = config['has_continuous_action_space']
        self.hidden_dim = config['hidden_dim']

        if self.has_continuous_action_space:
            self.action_dim = action_dim

        # actor
        if self.has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, self.hidden_dim),
                            nn.Tanh(),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.Tanh()
                            # nn.Linear(100, action_dim)
                            # nn.Tanh()
                        )
            self.mu = nn.Linear(self.hidden_dim, action_dim)
            self.logvar = nn.Linear(self.hidden_dim, action_dim)
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, self.hidden_dim),
                            nn.Tanh(),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.Tanh(),
                            nn.Linear(self.hidden_dim, action_dim),
                            nn.Softmax(dim=-1)
                        )  

    def act(self, state):

        if self.has_continuous_action_space:
            h = self.actor(state)
            action_mean = self.mu(h)
            action_var = torch.exp(0.5 * self.logvar(h))
            dist = Normal(action_mean, action_var)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            h = self.actor(state)
            action_mean = self.mu(h)
            action_var = torch.exp(0.5 * self.logvar(h))
            dist = Normal(action_mean, action_var)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        
        return action_logprobs


class ActorCritic(Actor):
    def __init__(self, state_dim, action_dim, config):
        super(ActorCritic, self).__init__(state_dim, action_dim, config)

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, self.hidden_dim),
                        nn.Tanh(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.Tanh(),
                        nn.Linear(self.hidden_dim, 1)
                    )

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            h = self.actor(state)
            action_mean = self.mu(h)
            action_var = torch.exp(0.5 * self.logvar(h))
            dist = Normal(action_mean, action_var)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class Metalearner:
    def __init__(self, state_dim, action_dim, config):

        self.has_continuous_action_space = config['has_continuous_action_space']

        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.K_epochs = config['K_epochs']
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, config).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': config['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': config['lr_critic']}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, config).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        # last_state = torch.squeeze(self.buffer.states[-1]).detach().to(device)
        # last_action = torch.squeeze(self.buffer.actions[-1]).detach().to(device)
        # _, last_value, _ = self.policy.evaluate(last_state, last_action)
        # rewards = [last_value.detach()]
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward) # 전체에 대한 discount reward 저장해두는 것
            
        # # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        old_logprobs = torch.sum(old_logprobs, dim = 1) # joint probability
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            logprobs = torch.sum(logprobs, dim = 1) # joint probability
            dist_entropy = torch.sum(dist_entropy, dim = 1)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   # discounted rewards - state_values
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss_clip = - torch.min(surr1, surr2)
            loss_VF = self.MseLoss(state_values, rewards)
            loss_S = - dist_entropy
            loss = loss_clip + 0.5*loss_VF + 0.01*loss_S
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

class Sampler:
    def __init__(self, state_dim, action_dim, config):

        self.has_continuous_action_space = config['has_continuous_action_space']

        self.gamma = config['gamma']
        self.fast_num_step = config['fast_num_step']
        
        self.buffer = RolloutBuffer()

        self.policy = Actor(state_dim, action_dim, config).to(device)
        self.optimizer = torch.optim.SGD(self.policy.actor.parameters(), lr=config['fast_lr'])

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return (action.detach().cpu().numpy().flatten(), 
                    action_logprob.detach().cpu().numpy().flatten())

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        # last_state = torch.squeeze(self.buffer.states[-1]).detach().to(device)
        # last_action = torch.squeeze(self.buffer.actions[-1]).detach().to(device)
        # _, last_value, _ = self.policy.evaluate(last_state, last_action)
        # rewards = [last_value.detach()]
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward) # 전체에 대한 discount reward 저장해두는 것
            
        # # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.fast_num_step):

            # Evaluating logprobability
            logprobs = self.policy.evaluate(states, actions)
            logprobs = torch.sum(logprobs, dim = 1) # joint probability

            # Finding Reinforce loss function
            loss = - rewards * logprobs
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
       

