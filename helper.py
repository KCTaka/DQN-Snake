import torch
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def random_rotate_tensor(images, set_angles=[0, 90, 180, 270]):
    '''
    Randomly rotate image (in torch tensor, not in numpy) to one of the set angles.
    '''

    angle = int(np.random.choice(set_angles))
    images = [TF.rotate(image, angle) for image in images]
    return images

def update_model_file(model, mode, episode, score_avg, replace=True):
    '''
    Update model file with the model so far. If replace is True, then it will remove all other models with the same mode.
    '''
    new_model_path = f"models/best_model_{mode}_e{episode}_s{score_avg:.2f}.pth"
    torch.save(model.state_dict(), new_model_path)
    
    if not replace:
        return
    
    old_files = glob.glob(f"models/best_model_{mode}_e*_s*.pth") # Get all files with the same mode
    for old_file in old_files:
        os.remove(old_file)
        
def load_model(model, mode, episode=None):
    # Construct the file pattern to search for models
    file_pattern = f"models/best_model_{mode}_e*_s*.pth"
    # Find all files that match the pattern
    model_files = glob.glob(file_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found for mode '{mode}'")
    
    if episode is not None:
        # Search for the specific episode
        specific_file = [f for f in model_files if f"_e{episode}_" in f]
        if specific_file:
            model_path = specific_file[0]
        else:
            raise FileNotFoundError(f"No model file found for episode {episode}")
    else:
        # Find the file with the highest episode number
        model_files.sort(key=lambda x: int(x.split('_e')[1].split('_')[0]), reverse=True)
        model_path = model_files[0]
    
    # Extract episode number and score average from the file name
    file_name = os.path.basename(model_path)
    episode = int(file_name.split('_e')[1].split('_')[0])
    score_avg = float(file_name.split('_s')[1].split('.pth')[0])
    
    # Load the model
    model.load_state_dict(torch.load(model_path, weights_only=False))
    return episode, score_avg

def view_image(image):
    '''
    Tool to see RGB image in numpy. Used for debugging purposes.
    '''
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    plt.imshow(image)
    plt.show()

def transform_into_array(shape):
    '''
    Used to turn shape into a numpy array. If int, then it will return a 1D array with the same value for both dimensions.
    '''
    
    if isinstance(shape, int):
        return np.array([shape, shape])
    elif len(shape) == 2:
        return np.array(shape)
    
    raise ValueError("Shape must be an integer or a tuple of two integers")

def shape_after_conv(input_shape, kernel_shape, stride, padding):
    '''
    Used to determine image dimensions after convolution, and maxpooling
    '''
    input_shape = transform_into_array(input_shape)
    kernel_shape = transform_into_array(kernel_shape)
    stride = transform_into_array(stride)
    padding = transform_into_array(padding)
    
    return (input_shape + 2 * padding - kernel_shape) // stride + 1

def normalize_angle(angle):
    '''
    Keep angle between -pi and pi; between -180 and 180 degrees
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    input_shape = (10, 50)
    kernel_shape = (3, 4)
    stride = (1, 2)
    padding = (1, 1)
    
    print(shape_after_conv(input_shape, kernel_shape, stride, padding))