"""
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)

"""

import gym
env = gym.make('FrozenLake8x8-v0')  # try for different environments
print(env.get_action_meanings())
env.reset()
env.render()

input("Press enter to exit")
env.close()
