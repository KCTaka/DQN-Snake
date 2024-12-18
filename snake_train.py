import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 
import torch.nn as nn

from collections import deque
from plotter import DynamicPlot

from helper import random_rotate_tensor, update_model_file


class ReplayMemory(Dataset):
    '''
    Provide a replay memory for the DQN model. Used in the DQN paper as D.
    If using action_ref and image as phi (game state) than rotation does not matter. For better generalization, we can rotate the image randomly.
    Random rotate is practically used for training CNN dqn mode "image" only.
    '''
    
    def __init__(self, maxlen=2000, random_rotate=False):
        super().__init__()
        self.maxlen = maxlen
        self.D = deque(maxlen=2000)
        
        self.random_rotate = random_rotate
        
    def append(self, transition):
        self.D.appendleft(transition)
    
    def __len__(self):
        return len(self.D)
    
    def __getitem__(self, idx):
        phi, a, reward, phi_next = self.D[idx]
        
        terminal = False
        if phi_next is None:
            phi_next = torch.zeros(phi.shape)
            terminal = True
            
            
        phi = torch.tensor(phi, dtype=torch.float32)
        phi_next = torch.tensor(phi_next, dtype=torch.float32)
            
        if self.random_rotate:
            phi, phi_next = random_rotate_tensor([phi, phi_next])
        
        return phi, a, reward, phi_next, terminal
    
class DQNTrainer:
    '''
    Training model based on the original DQN paper.
    Can be used to train any game other than snake but the game environment would require the same properties as the snake game. Should be specified in README.
    '''
    def __init__(self, 
                 model, 
                 game, 
                 get_state_from_sequence, 
                 batch_size=32, 
                 num_episodes = 15000, 
                 max_time_step = None, 
                 replay_maxlen = 2000, 
                 initial_epsilon = 0.6,
                 epsilon_linear_decay = 0.6/80,
                 min_epsilon = 0,
                 gamma=0.99, 
                 lr=0.001,
                 action_num = 3,
                 random_rotate = False,
                 ):
        '''
        epsilon_linear_decay: Linear decay of epsilon per episode (game of snake)
        '''
        
        self.model = model
        self.game = game
        self.get_state_from_sequence = get_state_from_sequence
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.max_time_step = max_time_step
        self.replay_memory = ReplayMemory(replay_maxlen, random_rotate=random_rotate)
        self.initial_epsilon = initial_epsilon
        self.epsilon_linear_decay = epsilon_linear_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.lr = lr
        self.action_num = action_num
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def get_action(self, phi, epsilon):
        '''
        Epsilon greedy policy.
        At probability epsilon, choose a random action.
        Otherwise, choose the action with the highest Q-value.
        '''
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_num)
        else:
            phi = torch.tensor(phi, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(phi)
            return torch.argmax(q_values).item()
        
    def update_model(self, sample_num):
        '''
        Backpropate specified in the DQN paper.
        sample_num: number of mini-batches to sample from the replay memory and backpropagate.
        
        Note:
        sample_num > 1 was not used in the original DQN paper. 
        If sample_num > 1, the iteration after the first will be y_hat calculated from the previous iteration model.
        I think it should accumulate the losses and backpropagate at the very end. I haven't tested that yet.
        '''
        batch_size = min(self.batch_size, len(self.replay_memory))
        dataloader = DataLoader(self.replay_memory, batch_size=batch_size, shuffle=True)
        
        for i, (phi, a, reward, phi_next, terminal) in enumerate(dataloader):     
            y = reward.clone().to(torch.float32)
            mask = terminal == False
            
            if mask.any():
                next_q_values = self.model(phi_next).detach()
                
                y[mask] += self.gamma*torch.max(next_q_values[mask], dim=1).values
            
            q_values = self.model(phi)
            y_hat = torch.gather(q_values, 1, a.unsqueeze(1)).squeeze(1)
            
            # Another way to calculate losses
            # target_q_values = q_values.clone()
            # target_q_values[range(len(target_q_values)), a] = y.reshape(-1)
            # loss = self.criterion(q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            
            if i >= sample_num:
                break
            
    def train(self, 
              mode = "Unspecified",
              num_frames = 4, 
              plot=True, 
              plot_episodes = 1, 
              gameplay=False, 
              gameplay_episodes=1, 
              score_maxlen=100,
              k = 1,
              save_plot=False,
              ):
        '''
        mode: should be either "simple", "sensor", or "image". Used to save the model file.
        num_frames = number of frames to consider for the state stored as s (sequence) in the DQN paper. (default was 4 in the paper)
        plot: If True, plot the training progress.
        plot_episodes: Plot the training progress every plot_episodes episodes.
        gameplay: If True, render the game while training.
        gameplay_episodes: Render the game every gameplay_episodes episodes. If larger than 5, it will quit pygame each time so it looks better.
        score_maxlen: Maximum number of scores to keep track of for the average score.
        k: Update the model every k time steps. (its best value varies depending on game as said in the paper)
        '''
        
        if plot:
            plotter = DynamicPlot(title="Training Plot", subplot_grid=(1, 2), set_size_inches=(10, 5))
        
        epsilon = self.initial_epsilon
        score_avg = []
        score_record = deque(maxlen=score_maxlen)
        
        max_avg_score = 0
        
        is_gameplaying = False
        for episode in range(self.num_episodes):
            # Initialize the game
            self.game.reset()
            self.game.update_image()
            
            # Initialize x: image and sequence s: {x}
            x = self.game.image
            x_0 = np.zeros_like(x)
            s = deque([x_0]*num_frames, maxlen=num_frames)
            s.appendleft(x)
            
            # Preprocess the state
            phi = self.get_state_from_sequence(s)
            
            # For rendering the game
            if is_gameplaying == False and (gameplay and episode % gameplay_episodes == 0):
                is_gameplaying = True
                self.game.init_pygame()
                print("Gameplay Started")
            
            epsilon = max(self.initial_epsilon - self.epsilon_linear_decay*episode, self.min_epsilon)
            score = 0
            t = 0
            while self.max_time_step is None or t < self.max_time_step:        
                if t % k != 0:
                    continue
                
                t += 1
                
                if is_gameplaying and episode % gameplay_episodes == 0:
                    self.game.render()
                    self.game.clock.tick(self.game.tick_rate)
                    # self.game.handle_events()
                
                a = self.get_action(phi, epsilon)
                
                self.game.do_action(a)
                reward = self.game.step()
                
                self.game.update_image() if not self.game.game_end else None # update_image doesn't work when snake is out of bounds. doesn't matter since the game is over.
                x_next = self.game.image
                
                s.appendleft(x_next)
                s_next = s # for better readability. It's a pointer so it doesn't matter.
                
                phi_next = self.get_state_from_sequence(s_next) if not self.game.game_end else None # If game is over, phi_next is None. Used to determine terminal state.
                transition = (phi, a, reward, phi_next)
                self.replay_memory.append(transition)
                
                phi = phi_next
                
                self.update_model(1)
                
                if self.game.game_end:
                    score = self.game.score
                    break
                
                if self.game.stop_game:
                    break
            
            # End rendering if gameplay is enabled
            if (is_gameplaying and gameplay_episodes > 5) or self.game.stop_game:
                self.game.quit_pygame()
                is_gameplaying = False
                print("Gameplay Ended")
            
            # Keep record of score
            score_record.append(score)
            
            # Update the model file if the average score is all-time high
            cur_score_avg = np.mean(score_record)
            if cur_score_avg > max_avg_score:
                print(f"Model saved with average score {cur_score_avg}")
                max_avg_score = cur_score_avg
                update_model_file(self.model, mode, episode, cur_score_avg, replace=True)
            
            # Status update for sanity check
            print(f"Episode {episode}\t|\tScore: {score}\t|\tEpsilon: {epsilon:.2f}\t|\tAvg Score: {cur_score_avg:.2f}\t|\tMax Avg Score: {max_avg_score:.2f}")
            
            # Plot the training progress if enabled
            if plot and episode % plot_episodes == 0:
                score_avg.append(np.mean(score_record))               
                mult_data_info = {
                    "mult_data": [score_avg, score_record],
                    "titles": [f"Average Score over max {len(score_record)} Episodes", f"Score for the past len {len(score_record)} Games"],
                    "xlabels": [f"{plot_episodes}th Episodes", f"Current - ({len(score_record)} - nth) Episodes"], 
                    "ylabels": ["Average Score", "Score"],
                }
                plotter.plot(**mult_data_info)
                if save_plot:
                    plotter.save_plot_image(f"plots/training_plot_{mode}.png")
            
            if self.game.stop_game:
                break

if __name__ == "__main__":
    pass
                
                
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                