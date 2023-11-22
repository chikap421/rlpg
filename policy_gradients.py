# %% [markdown]
# <a href="https://colab.research.google.com/drive/1WqOlp-uBfh7cjYezJwLLjy87UnngH48i" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # **Fall 2023 6.7900 HW 4 Policy Gradients**

# %% [markdown]
# ## (a) Setup
# 
# The following code sets up requirements, imports, and helper functions (you can ignore this).

# %%
!pip3 install -i https://test.pypi.org/simple/ sensorimotor-checker==0.0.8  &>/dev/null

# %%
!pip install gym-minigrid &>/dev/null

# %%
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import gym_minigrid
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm.notebook import tqdm
from gym_minigrid.envs.doorkey import DoorKeyEnv
import pandas as pd
import random
from sensorimotor_checker import hw2_tests
from gym_minigrid.wrappers import ImgObsWrapper

# %%
checker_policy_gradient = hw2_tests.TestPolicyGradients()

# %%
# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def preprocess_obss(obss, device=None):
    if isinstance(obss, dict):
        images = np.array([obss["image"]])
    else:
        images = np.array([o["image"] for o in obss])

    return torch.tensor(images, device=device, dtype=torch.float)

class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1

# %%
class Config:
    def __init__(self,
                score_threshold=0.93,
                discount=0.995,
                lr=1e-3,
                max_grad_norm=0.5,
                log_interval=10,
                max_episodes=1000,
                gae_lambda=0.95,
                use_critic=False,
                clip_ratio=0.2,
                target_kl=0.01,
                train_ac_iters=5,
                use_discounted_reward=True,
                entropy_coef=0.01,
                use_gae=False):

        self.score_threshold = score_threshold
        self.discount = discount
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.max_episodes = max_episodes
        self.use_critic = use_critic
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_ac_iters = train_ac_iters
        self.gae_lambda=gae_lambda
        self.use_discounted_reward=use_discounted_reward
        self.entropy_coef = entropy_coef
        self.use_gae = use_gae

# %%
env = DoorKeyEnv5x5()
env = ImgObsWrapper(DoorKeyEnv5x5())

# %%
x = env.reset()
print(x)


# %%
obs = env.reset()
print(obs)

img = obs[0]
print(env.observation_space)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


# %% [markdown]
# # Model
# 
# 

# %%
class ACModel(nn.Module):
    def __init__(self, num_actions, use_critic=False):
        super().__init__()
        self.use_critic = use_critic

        # Define actor's model
        self.image_conv_actor = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions)
        )

        # Define critic's model
        if self.use_critic:
            self.image_conv_critic = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )
            self.critic = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        conv_in = obs.transpose(1, 3).transpose(2, 3) # reshape into expected order

        dist, value = None, None

        x = self.image_conv_actor(conv_in)
        embedding = x.reshape(x.shape[0], -1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        if self.use_critic:
            y = self.image_conv_critic(conv_in)
            embedding = y.reshape(y.shape[0], -1)

            value = self.critic(embedding).squeeze(1)
        else:
            value = torch.zeros((x.shape[0], 1), device=x.device)

        return dist, value

# %% [markdown]
# # (b) REINFORCE
# 

# %% [markdown]
# Fill in `compute_discounted_return` code block below, to get the `returns` from a sequence of rewards. Each element of the `returns` array should be the sum of immediate reward, plus all discounted future rewards. The last element in `returns` has been filled out for you.
# 

# %%
def compute_discounted_return(rewards, discount, device=None):
    returns = torch.zeros(*rewards.shape, device=device)
    T = len(rewards)
    returns[T-1] = rewards[T-1]
    #### TODO: populate discounted reward trajectory ############
    for t in reversed(range(T-1)):
      returns[t] = rewards[t] + discount * returns[t+1]

    ######################################################################

    return returns

#### Test discounted return ####
checker_policy_gradient.test_compute_discounted_return(compute_discounted_return)

# %% [markdown]
# # Model Evaluation
# 
# The following code runs the model `acmodel` for one episode, and returns a dictionary with all the relevant (state, action, reward) from the rollout.  
# 
# It might be useful to review this code just to make sure you understand what's going on.

# %%
def collect_experiences(env, acmodel, args, device=None):
    """Collects rollouts and computes advantages.
    Returns
    -------
    exps : dict
        Contains actions, rewards, advantages etc as attributes.
        Each attribute, e.g. `exps['reward']` has a shape
        (self.num_frames, ...).
    logs : dict
        Useful stats about the training process, including the average
        reward, policy loss, value loss, etc.
    """


    MAX_FRAMES_PER_EP = 300
    shape = (MAX_FRAMES_PER_EP, )

    actions = torch.zeros(*shape, device=device, dtype=torch.int)
    values = torch.zeros(*shape, device=device)
    rewards = torch.zeros(*shape, device=device)
    log_probs = torch.zeros(*shape, device=device)
    obss = [None]*MAX_FRAMES_PER_EP

    obs, _ = env.reset()

    total_return = 0

    T = 0

    while True:
        # Do one agent-environment interaction

        preprocessed_obs = preprocess_obss(obs, device=device)

        with torch.no_grad():
            dist, value = acmodel(preprocessed_obs)
        action = dist.sample()[0]


        obss[T] = obs
        obs, reward, done, _, _ = env.step(action.item())


        # Update experiences values
        actions[T] = action
        values[T] = value
        rewards[T] = reward
        log_probs[T] = dist.log_prob(action)


        total_return += reward
        T += 1

        if done or T>=MAX_FRAMES_PER_EP-1:
            break

    discounted_reward = compute_discounted_return(rewards[:T], args.discount, device)
    exps = dict(
        obs = preprocess_obss([
            obss[i]
            for i in range(T)
        ], device=device),
        action = actions[:T],
        value  = values[:T],
        reward = rewards[:T],
        advantage = discounted_reward-values[:T],
        log_prob = log_probs[:T],
        discounted_reward = discounted_reward,
    )

    logs = {
        "return_per_episode": total_return,
        "num_frames": T
    }

    return exps, logs

# %%
def compute_policy_loss_reinforce(logps, returns):
    policy_loss = torch.tensor(0)
    #### TODO: complete policy loss, as the negative log-likelihood of actions weighted by returns ###
    policy_loss = -torch.mean(logps * returns)
    ############################################
    return policy_loss

#### Test policy loss for REINFORCE algorithm ####
checker_policy_gradient.test_compute_policy_loss_reinforce(compute_policy_loss_reinforce)

# %%
def update_parameters_reinforce(optimizer, acmodel, sb, args):

    logps, reward = None, None

    ### TODO: compute logps and reward from acmodel, sb['obs'] and sb['action'] ###
    obs = sb['obs']
    actions = sb['action']
    dist, _ = acmodel(obs)
    logps = dist.log_prob(actions)
    ##############################################################################################
    reward = sb['discounted_reward'] if args.use_discounted_reward else sb['reward']

    policy_loss = compute_policy_loss_reinforce(logps, reward)
    update_policy_loss = policy_loss.item()

    # Update actor-critic
    optimizer.zero_grad()
    policy_loss.backward()

    # Perform gradient clipping for stability
    for p in acmodel.parameters():
        if p.grad is None:
            print("Make sure you're not instantiating any critic variables when the critic is not used")
    update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in acmodel.parameters()) ** 0.5
    torch.nn.utils.clip_grad_norm_(acmodel.parameters(), args.max_grad_norm)
    optimizer.step()

    # Log some values
    logs = {
        "policy_loss": update_policy_loss,
        "grad_norm": update_grad_norm
    }

    return logs

# %% [markdown]
# Now, let's try to run our implementation.  The following experiment harness is written for you, and will run sequential episodes of policy gradients until `args.max_episodes` timesteps are exceeded or the rolling average reward (over the last 100 episodes is greater than `args.score_threshold`. It is expected to get highly variable results, and we'll visualize some of this variability at the end.
# 
# The method accepts as arguments a `Config` object `args`, and a `parameter_update` method (such as `update_parameters_reinforce`).

# %%
def run_experiment(args, parameter_update, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DoorKeyEnv5x5()

    acmodel = ACModel(env.action_space.n, use_critic=args.use_critic)
    acmodel.to(device)

    is_solved = False

    SMOOTH_REWARD_WINDOW = 50

    pd_logs, rewards = [], [0]*SMOOTH_REWARD_WINDOW

    optimizer = torch.optim.Adam(acmodel.parameters(), lr=args.lr)
    num_frames = 0

    pbar = tqdm(range(args.max_episodes))
    for update in pbar:
        exps, logs1 = collect_experiences(env, acmodel, args, device)
        logs2 = parameter_update(optimizer, acmodel, exps, args)

        logs = {**logs1, **logs2}

        num_frames += logs["num_frames"]

        rewards.append(logs["return_per_episode"])

        smooth_reward = np.mean(rewards[-SMOOTH_REWARD_WINDOW:])

        data = {'episode':update, 'num_frames':num_frames, 'smooth_reward':smooth_reward,
                'reward':logs["return_per_episode"], 'policy_loss':logs["policy_loss"]}

        if args.use_critic:
            data['value_loss'] = logs["value_loss"]

        pd_logs.append(data)

        pbar.set_postfix(data)

        # Early terminate
        if smooth_reward >= args.score_threshold:
            is_solved = True
            break

    if is_solved:
        print('Solved!')

    return pd.DataFrame(pd_logs).set_index('episode')

# %% [markdown]
# ## Run REINFORCE
# 
# This should generally converge. If it doesn't, try re-running the cell with a few different seeds to make sure it's not an error. It might also help to increase the `max_iter` argumenet below. Also this is expected to take quite a few minutes.

# %%
args = Config(use_discounted_reward=True)
# args = Config(max_iter=2000)
df = run_experiment(args, update_parameters_reinforce)

df.plot(x='num_frames', y=['reward', 'smooth_reward'])

# %% [markdown]
# # (c) Vanilla Policy Gradients
# 

# %%
def compute_policy_loss_with_baseline(logps, advantages):


    ### TODO: implement the policy loss, as the negative log-likelihood of actions weighted by advantages ######
    policy_loss = -(logps *advantages).mean()
    ##################################################

    return policy_loss

#### Test discounted return ####
checker_policy_gradient.test_compute_policy_loss_with_baseline(compute_policy_loss_with_baseline)

# %%
def update_parameters_with_baseline(optimizer, acmodel, sb, args):

    def _compute_value_loss(values, returns):
        value_loss = F.mse_loss(values, returns)

        ### TODO: implement the value loss #######

        ##################################################

        return value_loss

    ### TODO: populate the policy and value loss computation fields using acmodel, sb['obs'], sb['action], and sb['discounted_reward']
    ### For the advantage term, use sb['advantage'].
    obs = sb['obs']
    actions = sb['action']
    dist, values = acmodel(obs)
    logps = dist.log_prob(actions)
    advantages = sb['advantage']
    discounted_rewards = sb['discounted_reward']

    ####################################################################################################

    policy_loss = compute_policy_loss_with_baseline(logps, advantages)
    value_loss = _compute_value_loss(values, discounted_rewards)
    loss = policy_loss + value_loss

    update_policy_loss = policy_loss.item()
    update_value_loss = value_loss.item()

    # Update actor-critic
    optimizer.zero_grad()
    loss.backward()
    update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in acmodel.parameters()) ** 0.5
    torch.nn.utils.clip_grad_norm_(acmodel.parameters(), args.max_grad_norm)
    optimizer.step()

    # Log some values

    logs = {
        "policy_loss": update_policy_loss,
        "value_loss": update_value_loss,
        "grad_norm": update_grad_norm
    }

    return logs

# %% [markdown]
# ## Run VPG
# 
# If you did everything right, you should be able to run the below cell to run the vanilla policy gradients implementation with baseline.  This should be somewhat more stable than without the baseline, and likely converge faster.
# 

# %%
args = Config(use_critic=True)
df_baseline = run_experiment(args, update_parameters_with_baseline)

df_baseline.plot(x='num_frames', y=['reward', 'smooth_reward'])


