import numpy as np
import cv2 as cv
from helper import normalize_angle, view_image

'''
Used as phi function in the DQN paper. This function will take the image sequence and return the state of the game, phi.
The state of the game will be used as input to the neural network.
'''

def get_game_info_from_sequence(image_sequence):
    '''
    This function will take the image sequence and return the game information needed to determine the state of the game.
    Should give you a dictionary with:
    - head_pos: Position of the head of the snake
    - snake_pos: Position of all the segments of the snake
    - food_pos: Position of the food
    - action_dir: Direction of the snake
    - grid_size: Size of the grid
    
    Note that its not that robust. It won't work if snake doesn't move. It assumes it always moves because the model doesn't have the option to stop. 
    Also image_sequence in for the snake game used binary chanelled images specified in update_image function of snake_game.py
     - binary channel image: first channel is the snake, second channel is the food
    '''
    
    if len(image_sequence) < 3:
        # I tried to do it with 2 images, but I think its not possible given how too simple the snake game is. 
        raise ValueError("Image sequence must have at least 3 images to determine direction of snake")
    
    prev_prev_image = image_sequence[2]
    prev_image = image_sequence[1]
    curr_image = image_sequence[0]
    grid_size = np.array(curr_image[0].shape)
    
    image_diff = curr_image - prev_image
    prev_image_diff = prev_image - prev_prev_image
    
    snake_pos = np.argwhere(curr_image[0] == 1)
    food_pos = np.argwhere(curr_image[1] == 1).squeeze()
    
    head_pos = np.argwhere(image_diff[0] == 1).squeeze()
    prev_head_pos = np.argwhere(prev_image_diff[0] == 1).squeeze()
    
    if prev_head_pos.size == 0:
        prev_head_pos = head_pos - np.array([0, -1])

    action_dir = head_pos - prev_head_pos
    
    game_info = {
        "head_pos": head_pos,
        "snake_pos": snake_pos,
        "food_pos": food_pos,
        "action_dir": action_dir,
        "grid_size": grid_size
    }
    
    return game_info

def convert_image_to_game_image(image, window_size):
    '''
    I have to do this weird thing here where I convert the simple image representation into the image seen in gameplay so that the CNN has a more accurate representation of the game state.
    We set it so that the snake is green and the food is red. We can change this to what we want. I think CNN works better this way. This is more DQN paper accurate.
    '''
    green_color_code = (np.array([0, 1, 0])*255)
    red_color_code = (np.array([1, 0, 0])*255)
    
    window_size = np.array(window_size)
    grid_size = np.array(image[0].shape)
    cell_size = window_size // grid_size
    
    green_tile = np.tile(green_color_code, (*cell_size, 1))
    red_tile = np.tile(red_color_code, (*cell_size, 1))
    
    game_image = np.zeros((*window_size, 3))
    
    snake_pos = np.argwhere(image[0] == 1)
    food_pos = np.argwhere(image[1] == 1)
    
    if food_pos.size == 0:
        return game_image.astype(np.uint8)
    
    food_pos = food_pos.squeeze()

    for segment in snake_pos:
        segment_start = segment * cell_size
        segment_end = segment_start + cell_size
        game_image[segment_start[0]:segment_end[0], segment_start[1]:segment_end[1]] = green_tile
    
    food_start = food_pos * cell_size
    food_end = food_start + cell_size
    game_image[food_start[0]:food_end[0], food_start[1]:food_end[1]] = red_tile
    
    return game_image.astype(np.uint8)

def sensor(angle, action_dir, head_pos, snake_pos, food_pos, grid_size):
    '''
    Image this sensor as a laser beam that shoots out from the head of the snake at an angle based on snake's perspective and returns the distance and type of object it hits.
    Might not be as accurate with small grid sizes but I decided to stop thinking about it (distance is calculated from the center of the cell blocks, so snake right next to a block still counts as distance 1). 
    '''
    head_angle = np.arctan2(action_dir[1], action_dir[0])
    sensor_angle = normalize_angle(head_angle + angle)
    sensor_dir = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
    
    for segment in snake_pos:
        dist = np.linalg.norm(segment - food_pos)
        pos = head_pos + sensor_dir*dist
        pos = np.round(pos).astype(int)
        if np.array_equal(pos, food_pos):
            return dist, "body"
        
    dist = np.linalg.norm(head_pos - food_pos)
    pos = head_pos + sensor_dir*dist
    pos = np.round(pos).astype(int)
    if np.array_equal(pos, food_pos):
        return dist, "food"
    
    wall_dists_rb = grid_size - head_pos
    wall_dists_tl = head_pos + np.ones(2, dtype=int)
    
    corner_angles = [np.arctan2(-wall_dists_tl[1], -wall_dists_tl[0]), 
                     np.arctan2(-wall_dists_tl[1], wall_dists_rb[0]), 
                     np.arctan2(wall_dists_rb[1], wall_dists_rb[0]), 
                     np.arctan2(wall_dists_rb[1], -wall_dists_tl[0])]
    
    for i in range(len(corner_angles)):
        prev_i = (i-1)%len(corner_angles)
        corner_angle_diff = normalize_angle(corner_angles[prev_i] - corner_angles[i])
        prev_corner_angle = corner_angles[i] + corner_angle_diff
        if prev_corner_angle <= sensor_angle <= corner_angles[i]:
            if i == 0:
                wall_angle = -np.pi
                wall_dist = wall_dists_tl[0]
            elif i == 1:
                wall_angle = -np.pi/2
                wall_dist = wall_dists_tl[1]
            elif i == 2:
                wall_angle = 0
                wall_dist = wall_dists_rb[0]
            elif i == 3:
                wall_angle = np.pi/2
                wall_dist = wall_dists_rb[1]
            
            angle_diff = np.abs(normalize_angle(sensor_angle - wall_angle))
            dist = wall_dist/np.cos(angle_diff)
            return dist, "wall"
        
    wall_angle = -np.pi/2
    wall_dist = wall_dists_tl[0]
    angle_diff = np.abs(normalize_angle(sensor_angle - wall_angle))
    dist = wall_dist/np.cos(angle_diff)
    return dist, "wall"
    
           
def sensor_state(image_sequence, angles = [0]):
    '''
    Returns the state of the game based on the sensors (like LiDAR) at specified set of given angles.
    It returns a 1d array game state, phi, where the first element is the distance to the object and the second element is the type of object (0 for dangerous object, 1 for food).  
    '''
    
    game_info = get_game_info_from_sequence(image_sequence)
    head_pos = game_info["head_pos"]
    snake_pos = game_info["snake_pos"]
    food_pos = game_info["food_pos"]
    action_dir = game_info["action_dir"]
    grid_size = game_info["grid_size"]
    
    game_state = np.zeros(2*len(angles)) 
    
    for i, angle in enumerate(angles):
        dist, obj = sensor(angle, action_dir, head_pos, snake_pos, food_pos, grid_size)
        
        game_state[2*i] = dist
        game_state[2*i + 1] = obj == "food"
    
    return game_state

def binary_state(image_sequence):
    '''
    This state is a state used in a DQN snake tutorial on youtube and specified in README.md. 
    It returns a 1d array game state, phi, where:
    index 0: 1 if danger infront of snake
    index 1: 1 if danger to the right of snake
    index 2: 1 if danger to the left of snake
    index 3: 1 if snake is moving up
    index 4: 1 if snake is moving down
    index 5: 1 if snake is moving left
    index 6: 1 if snake is moving right
    index 7: 1 if food is to the left of the snake
    index 8: 1 if food is to the right of the snake
    index 9: 1 if food is above the snake
    index 10: 1 if food is below the snake
    '''
    game_info = get_game_info_from_sequence(image_sequence)
    head_pos = game_info["head_pos"]
    snake_pos = game_info["snake_pos"]
    food_pos = game_info["food_pos"]
    action_dir = game_info["action_dir"]
    grid_size = game_info["grid_size"]
    
    game_state = np.zeros(11)
    
    danger_straight, obj_straight = sensor(0, action_dir, head_pos, snake_pos, food_pos, grid_size)
    danger_right, obj_right = sensor(np.pi/2, action_dir, head_pos, snake_pos, food_pos, grid_size)
    danger_left, obj_left = sensor(-np.pi/2, action_dir, head_pos, snake_pos, food_pos, grid_size)
    
    game_state[0] = np.isclose(danger_straight, 1) and obj_straight != "food"
    game_state[1] = np.isclose(danger_right, 1) and obj_right != "food"
    game_state[2] = np.isclose(danger_left, 1) and obj_left != "food"
    
    game_state[3] = np.array_equal(action_dir, np.array([-1, 0]))
    game_state[4] = np.array_equal(action_dir, np.array([1, 0]))
    game_state[5] = np.array_equal(action_dir, np.array([0, -1]))
    game_state[6] = np.array_equal(action_dir, np.array([0, 1]))
    
    game_state[7] =  food_pos[0] < head_pos[0]
    game_state[8] =  food_pos[0] > head_pos[0]
    game_state[9] =  food_pos[1] < head_pos[1]
    game_state[10] =  food_pos[1] > head_pos[1]
    
    return game_state

def image_state(image_sequence, window_size = (168, 168)):
    '''
    DQN paper accuate state representation for CNN model.
    Creates an game image from our custom binary chanelled image and preprocess as specified in the DQN paper.
    Returns a 4x84x84 numpy array.
    '''
    game_image_sequence = []
    for image in image_sequence:
        game_image = convert_image_to_game_image(image, window_size)
        gray_image = cv.cvtColor(game_image, cv.COLOR_RGB2GRAY)
        game_image = cv.pyrDown(gray_image) # We want size (84, 84)
        # game_image = cv.resize(gray_image, (84, 84))
        
        game_image_sequence.append(game_image)
        
    # cv.imshow("Game Image", game_image_sequence[0])
    # print("Game Image Shape:", game_image_sequence[0].shape)
    
    game_image_sequence = np.array(game_image_sequence)
    
    return game_image_sequence
        
    
    