import pygame
from itertools import product
import numpy as np

class SnakeGame():
    def __init__(self, window_size, grid_size, default_reward = 0, food_reward = 1, death_reward = -1, norm_spawn = False, tick_rate=10):
        
        self.window_size = np.array((window_size, window_size), dtype=int) if isinstance(window_size, int) else np.array(window_size, dtype=int)
        self.grid_size = np.array((grid_size, grid_size), dtype=int) if isinstance(grid_size, int) else np.array(grid_size, dtype=int)
        self.cell_size = self.window_size // self.grid_size
        self.tick_rate = tick_rate
        
        self.default_reward = default_reward    
        self.food_reward = food_reward
        self.death_reward = death_reward
        
        self.norm_spawn = norm_spawn
        
        # Intialize snake
        random_pos = self.generate_norm_pos() if norm_spawn else self.generate_rand_pos()
        self.snake = [random_pos]
        self.action = np.zeros(2)
        
        # Initialize food position
        self.food_pos = self.generate_norm_pos() if norm_spawn else self.generate_food_pos(self.snake)
        
        # Initialize game state grid_size array
        self.image = np.zeros((2, self.grid_size[1], self.grid_size[0]), dtype=int)
        self.update_image()
        
        self.score = 0
        self.frame_iter = 0
        
        self.game_end = False
        self.stop_game = False
        
    def init_pygame(self):
        self.screen = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
        self.clock = pygame.time.Clock()
        
    def quit_pygame(self):
        pygame.quit()
        
    def generate_rand_pos(self):
        random_x_pos = np.random.randint(0, self.grid_size[0], 1, dtype=int)
        random_y_pos = np.random.randint(0, self.grid_size[1], 1, dtype=int)
        random_pos = np.concatenate([random_x_pos, random_y_pos], axis=0)
        return random_pos
    
    def generate_avail_pos(self, excluded_pos):
        all_pos_set = set(product(range(self.grid_size[0]), range(self.grid_size[1])))
        excluded_pos_set = set(map(tuple, excluded_pos))
        avail_pos = list(all_pos_set - excluded_pos_set)
        return avail_pos
    
    def generate_norm_pos(self):
        random_pos = np.round(np.random.multivariate_normal(np.array(self.grid_size)/2, np.diag(np.array(self.grid_size)/8)))
        random_pos = np.array([np.clip(num, 0, self.grid_size[i]-1) for i, num in enumerate(random_pos)], dtype=int)
        return random_pos
    
    def generate_food_pos(self, excluded_pos=[]):
        avail_pos = self.generate_avail_pos(excluded_pos)
        random_index = np.random.randint(0, len(avail_pos))
        return avail_pos[random_index]
        
    def is_collided(self, pos):
        # Check if pos is within the grid
        if np.any(np.greater_equal(pos, self.grid_size)) or np.any(np.less(pos, 0)):
            return True
        
        # Check if pos is within the snake
        if any(np.array_equal(pos, segment) for segment in self.snake[1:]):
            return True
        return False
    
    def step(self):
        self.frame_iter += 1
        # Initialize reward
        reward = self.default_reward
        
        # Move the snake
        new_head = self.snake[0] + self.action
        self.snake = [new_head] + self.snake if not np.array_equal(self.action, np.zeros(2, dtype=int)) else self.snake
        
        # Check if game ends due to collision
        if self.is_collided(new_head) or self.frame_iter > 100*len(self.snake):
            reward = self.death_reward
            self.game_end = True
            
        # Generate new food if eaten
        if np.array_equal(new_head, self.food_pos):
            reward = self.food_reward
            self.score += 1
            self.food_pos = self.generate_food_pos(self.snake)
        elif not np.array_equal(self.action, np.zeros(2, dtype=int)):
            self.snake.pop()
        
        return reward
                
    def update_image(self):
        '''
        This image is a representative of the state of the game. Not necessarily game state. Would be the x in the DQN paper.
        This image is not RGB but a binary channeled image. The first channel is the snake and the second channel is the food.
        '''
        
        
        self.image = np.zeros_like(self.image)
        for segment in self.snake:
            self.image[0][segment[1], segment[0]] = 1
            
        self.image[1][self.food_pos[1], self.food_pos[0]] = 1
        
        return self.image
    
    def get_game_image(self): # Not used as I chose not to render the game when training
        '''
        Outputs rendered game image as a numpy array. Ended up not using it because I wanted to train without rendering each frame. 
        '''
        return pygame.surfarray.array3d(self.screen)
            
    def render(self):
        self.screen.fill((0, 0, 0)) # Black background
        
        # Draw snake
        for segment in self.snake:
            segment_pos = segment * self.cell_size
            pygame.draw.rect(self.screen, (0, 255, 0), (*segment_pos, *self.cell_size))
        
        # Draw food
        food_pos = self.food_pos * np.array(self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), (*food_pos, *self.cell_size))
        
        pygame.display.flip()
            
    def handle_events(self, user_input=False):
        '''
        Handle user input and quit events. User input is only handled when user_input is True.
        '''
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_game = True
            
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.game_end = True
                
                if not user_input:
                    continue
                
                if event.key == pygame.K_UP:
                    self.action = np.array([0, -1]) if not np.array_equal(self.action, np.array([0, 1])) else self.action
                if event.key == pygame.K_DOWN:
                    self.action = np.array([0, 1]) if not np.array_equal(self.action, np.array([0, -1])) else self.action
                if event.key == pygame.K_RIGHT:
                    self.action = np.array([1, 0]) if not np.array_equal(self.action, np.array([-1, 0])) else self.action
                if event.key == pygame.K_LEFT:
                    self.action = np.array([-1, 0]) if not np.array_equal(self.action, np.array([1, 0])) else self.action
                    
    def do_action(self, action):
        '''
        Choose between do_action_ref or do_action_intertial. This function is called in the trianing. 
        Action_ref is primarly used for all modes. 
        Action_intertial was used for testing and was tested on CNN "image" mode but failed to learn for some reason.
        '''
        self.do_action_ref(action)                
    
    def do_action_ref(self, action):    
        '''
        Action is based on the snake's reference frame / direction of snake. Either it goes left, right or forward from the snake's perspective.
        '''
        
        if action == 0: # Left
            self.action = np.array([[0, 1], [-1, 0]], dtype=int) @ self.action if not np.array_equal(self.action,np.zeros(2, dtype=int)) else np.array([-1, 0])
            return 
        elif action == 1: # Forward
            self.action = self.action if not np.array_equal(self.action, np.zeros(2, dtype=int)) else np.array([0, -1])
            return
        elif action == 2: # Right
            self.action = np.array([[0, -1], [1, 0]], dtype=int) @ self.action if not np.array_equal(self.action, np.zeros(2, dtype=int)) else np.array([1, 0])
            return
        
        raise ValueError("Invalid action")
    
    def do_action_intertial(self, action):
        '''
        Action is based on the fixed world frame. In other words:
        action 0 will always move the snake up
        action 1 will always move the snake left
        action 2 will always move the snake down
        action 3 will always move the snake right
        '''
        
        if action == 0: # Up
            self.action = np.array([0, -1]) if not np.array_equal(self.action, np.array([0, 1])) else self.action
        if action == 1: # Left
            self.action = np.array([-1, 0]) if not np.array_equal(self.action, np.array([1, 0])) else self.action
        if action == 2: # Down
            self.action = np.array([0, 1]) if not np.array_equal(self.action, np.array([0, -1])) else self.action
        if action == 3: # Right
            self.action = np.array([1, 0]) if not np.array_equal(self.action, np.array([-1, 0])) else self.action
            
    def reset(self):
        '''
        Reset values to start new game/episode
        '''
        random_pos = self.generate_norm_pos() if self.norm_spawn else self.generate_rand_pos()
        self.snake = [random_pos]
        
        self.food_pos = self.generate_norm_pos() if self.norm_spawn else self.generate_food_pos(self.snake)
        
        self.action = np.zeros(2)
        self.image = np.zeros_like(self.image)
        self.score = 0
        self.frame_iter = 0
        
        self.game_end = False
            
    def run(self):
        '''
        Test the game using user controls and debugging
        '''
        while not self.game_end:
            self.handle_events(user_input=True)
            reward = self.step()
            self.update_image() if not self.game_end else None    
            self.render()
            self.clock.tick(self.tick_rate)
            
            print("Reward:", reward)
            print(f"Game State:\n{self.image}")
            print(f"Position:\t{self.snake[0]})")
            # game_image = self.get_game_image()
            # view_image(game_image)
            
            if self.game_end:
                self.reset()
            
            if self.stop_game:
                pygame.quit()
                break

if __name__ == "__main__":
    game = SnakeGame(
        window_size=400, 
        grid_size=10, 
        food_reward=10,
        death_reward=-10,
        tick_rate=3
    )
    game.init_pygame()
    game.run()
    
        
        
        
        