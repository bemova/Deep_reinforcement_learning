import gym
import numpy as np
from gym_mountain_car.tile_coding.tilecoder import tilecoder

env = gym.make("MountainCar-v0")

tile = tilecoder(env, 4, 18)
theta = np.random.uniform(-0.001, 0, size=(tile.n))
# Custom alpha learned to generalize based upon number of tilings
alpha = (.1/ tile.numTilings)*3.2
# Discounting not needed since reward gives -1 reward each time step
gamma = 1
numEpisodes = 10000
stepsPerEpisode = 200
rewardTracker = []
render = False
solved = False

for episodeNum in range(1, numEpisodes + 1):
    G = 0
    state = env.reset()
    for step in range(stepsPerEpisode):
        if render:
            env.render()
        F = tile.getFeatures(state)
        Q = tile.getQ(F, theta)
        action = np.argmax(Q)
        state2, reward, done, info = env.step(action)
        G += reward
        delta = reward - Q[action]
        if done:
            theta += np.multiply((alpha * delta), tile.oneHotVector(F, action))
            rewardTracker.append(G)
            if episodeNum % 100 == 0:
                print("Total Episodes = {}    Episode Reward = {}    Average Reward = {:04.1f}" \
                      .format(episodeNum, G, np.mean(rewardTracker)))
            break
        Q = tile.getQ(tile.getFeatures(state2), theta)
        delta += gamma * np.max(Q)
        theta += np.multiply((alpha * delta), tile.oneHotVector(F, action))
        state = state2

    if not solved:
        if episodeNum > 100:
            if sum(rewardTracker[episodeNum - 100:episodeNum]) / 100 >= -110:
                print('Solved in {} Episodes'.format(episodeNum))
                render = True
                solved = True