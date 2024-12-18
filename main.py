import numpy as np

from snake_game import SnakeGame
from snake_model import FCNN, MultiCNN, CNN
from snake_train import DQNTrainer
from snake_states import sensor_state, binary_state, image_state

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train_dqn(game, mode="simple"):
    if mode == "simple":
        model = FCNN(11, [128], 3)
        get_state_from_sequence = binary_state
        random_rotate = False
        
    elif mode == "sensor":
        sensors = np.linspace(-np.pi*5/4, np.pi*5/4, 13)
        model = FCNN(len(sensors)*2, [256, 128], 3)
        get_state_from_sequence = lambda x: sensor_state(x, sensors)
        random_rotate = False
        
    elif mode == "image":
        kwargs = {
            "conv_and_relu1": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 4, "padding": 0},
            "conv_and_relu2": {"in_channels": 16, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
            "fc1": {"in_features": 32*9*9, "out_features": 256},
            "fc2": {"in_features": 256, "out_features": 3},
        }
        model = CNN(input_shape=(3, 84, 84), **kwargs)
        get_state_from_sequence = image_state
        random_rotate = True
        
    trainer = DQNTrainer(model=model,
                    game=game,
                    get_state_from_sequence=get_state_from_sequence,
                    batch_size=32,
                    num_episodes=1_000_000, 
                    max_time_step=None,
                    replay_maxlen=1_000_000,
                    initial_epsilon=1,
                    epsilon_linear_decay=1/5_000,
                    min_epsilon=0.05,
                    gamma=0.95,
                    lr=5e-5,
                    action_num=3,
                    random_rotate=random_rotate,
                    )
    
    trainer.train(
        mode = mode,
        num_frames=4,
        plot=True,
        plot_episodes=100,
        gameplay=False,
        gameplay_episodes=10,
        score_maxlen=100,
        k = 1,
        save_plot=True,
    )

if __name__ == "__main__":
    game = SnakeGame(
        window_size=400, 
        grid_size=8, 
        default_reward=-0.1,
        food_reward=1,
        death_reward=-1,
        tick_rate=60,
    )
    
    train_dqn(game, mode="image")
     