import torch
import numpy as np
from collections import deque

from snake_model import FCNN, MultiCNN, CNN
from snake_game import SnakeGame
from snake_states import sensor_state, binary_state, image_state

from helper import view_image, load_model


def test_agent(model, game, get_state_from_sequence, num_frames = 4, num_games = 1000):
    '''
    Test the agent with the given model and game environment.
    num_frames: Number of frames to consider for the state stored as s (sequence) in the DQN paper.
    num_games: Number of games to play before quitting automatically
    '''
    
    for i in range(num_games):
        game.reset()
        game.update_image()
        
        x = game.image
        x_0 = np.zeros_like(x)
        s = deque([x_0]*num_frames, maxlen=num_frames)
        s.appendleft(x)
        
        phi = get_state_from_sequence(s)
        
        score = 0
        while True:
            game.handle_events(user_input=False)
            game.render()
            game.clock.tick(game.tick_rate)
            
            
            q_values = model(torch.tensor(phi, dtype=torch.float32).unsqueeze(0))
            a = torch.argmax(q_values).item()
            
            game.do_action(a)
            
            reward = game.step()
            
            game.update_image() if not game.game_end else None
            s.appendleft(game.image)
            phi = get_state_from_sequence(s) if not game.game_end else None
            
            if game.game_end:
                score = game.score
                break
            
            if game.stop_game:
                break
            
        print(f"Game {i+1} Score: {score}")
        
        if game.stop_game:
            break
        
    game.quit_pygame()
    
def test_model(mode = "simple"):
    '''
    Model is tested for each types of DQN.
    mode: simple, sensor, image
    architecture of each is specified in the snake_model.py. Make sure they are the same as the one used for training. Too lazy to implement a check for that.
    
    Press q to restart game.
    Quit pygame to end testing.
    '''
    
    game = SnakeGame(
        window_size=400, 
        grid_size=8, 
        default_reward=-0.1,
        food_reward=2,
        death_reward=-1,
        tick_rate=10,
    )
    
    game.init_pygame()
    
    if mode == "simple":
        model = FCNN(11, [128], 3)
        get_state_from_sequence = binary_state
        
    elif mode == "sensor":
        sensors = np.linspace(-np.pi*5/4, np.pi*5/4, 13)
        model = FCNN(len(sensors)*2, [256, 128], 3)
        get_state_from_sequence = lambda x: sensor_state(x, sensors)
    
    elif mode == "image":
        kwargs = {
            "conv_and_relu1": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 4, "padding": 0},
            "conv_and_relu2": {"in_channels": 16, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
            "fc1": {"in_features": 32*9*9, "out_features": 256},
            "fc2": {"in_features": 256, "out_features": 3},
        }
        model = CNN(input_shape=(3, 84, 84), **kwargs)
        get_state_from_sequence = image_state
        
    episode, score_avg = load_model(model, mode)
    print(f"Model mode {mode} loaded from episode {episode} with average score {score_avg}")
    
    test_agent(model, game, get_state_from_sequence, num_frames=4, num_games=10)


if __name__ == "__main__":
    mode = "image" # simple, sensor, image
    test_model(mode)