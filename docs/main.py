import pygame
import asyncio
import numpy as np
from pyodide.ffi import to_js
from js import getAIMove

# Import game code
from snake_game import SnakeGame
from snake_states import image_state

ai_controlled = False

async def main():
    pygame.init()
    game = SnakeGame(
        window_size=400,
        grid_size=8,
        default_reward=-0.1,
        food_reward=2,
        death_reward=-1,
        tick_rate=10
    )
    game.init_pygame()
    screen = game.screen

    num_frames = 4
    s = None

    while True:
        game.handle_events(user_input=False)
        screen.fill((0,0,0))

        if s is None or game.game_end:
            game.reset()
            game.update_image()
            x = game.image
            x0 = np.zeros_like(x)
            from collections import deque
            s = deque([x0]*num_frames, maxlen=num_frames)
            s.appendleft(x)

        if ai_controlled:
            phi = image_state(s)
            flat = phi.flatten().tolist()
            move = await getAIMove(to_js(flat))
            game.do_action(move)
        else:
            game.render()
            await asyncio.sleep(0)
            # manual input handled in handle_events

        reward = game.step()
        if not game.game_end:
            game.update_image()
            s.appendleft(game.image)

        game.render()
        pygame.display.flip()
        await asyncio.sleep(0)

# Function to be called by JavaScript

def toggle_ai(is_ai_on):
    global ai_controlled
    ai_controlled = is_ai_on
