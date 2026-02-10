import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
from torch import nn
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR,exist_ok=True)
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self,hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets=yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set =hyperparameter_set

        self.env_id             = hyperparameters['env_id']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g    = hyperparameters['discount_factor_g']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fcl_nodes          = hyperparameters['fcl_nodes']
        self.max_episodes       = hyperparameters.get('max_episodes', 1000)
        self.env_make_params    = hyperparameters.get('env_make_params',{})
 
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
    
    def run(self, is_training=True, render=False):
        # Utilise self.env_id au lieu de "CartPole-v1" en dur
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states,num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states,num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0
            best_reward = -99999999

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        else :
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            epsilon = 0  # Pas d'exploration en mode test
            memory = None
            target_dqn = None

        # Ajoute une limite au nombre d'épisodes
        for episode in range(self.max_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated=False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):

                if is_training and random.random() < epsilon :
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else :
                    with torch.no_grad():
                    # 1 dimension => 2D
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()


                # Processing:
                newState, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                newState = torch.tensor(newState, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                if is_training:
                    memory.append((state,action,newState, reward, terminated))
                    step_count+=1

                state=newState
            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"New best reward {episode_reward:0.1f}(best: {best_reward})"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

                epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Affiche la progression tous les 50 épisodes
                if (episode + 1) % 50 == 0:
                    print(f"Episode {episode + 1}/{self.max_episodes}, Reward: {episode_reward:.1f}, Epsilon: {epsilon:.3f}")
            else:
                # Mode test : affiche les résultats
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}/{self.max_episodes}, Reward: {episode_reward:.1f}")

        # Sauvegarde le graphique à la fin
        if is_training:
            self.save_graph(rewards_per_episode, epsilon_history)
            print(f"Graphique sauvegardé: {self.GRAPH_FILE}")
        else:
            avg_reward = np.mean(rewards_per_episode)
            print(f"\nMoyenne des récompenses sur {self.max_episodes} épisodes: {avg_reward:.1f}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)
        
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
        
        # Calcule current_q une seule fois et correctement
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0,x-99):(x+1)])
        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description = 'train our test model')
    parser.add_argument('hyperparameters', help='Name of hyperparameter set in hyperparameters.yml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)
    
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)