
# to play game use below in terminal:
# C:\Users\Saumya\anaconda3\anaconda3\envs\nengo_env\python.exe flappy_bird.py
import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# creating env
env = gym.make("FlappyBird-v0", render_mode = "human")
state,_ = env.reset()
done = False

# Intialize Pygame Keyboard
pygame.init()
screen = pygame.display.get_surface()

while not done:
    action = 0  # default 0 is no flap, 1 is flap
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
             if event.key == pygame.K_SPACE:
                 action = 1 # flap
    state, reward, done, truncated, _ = env.step(action)
    env.render()
env.close()
pygame.quit()
