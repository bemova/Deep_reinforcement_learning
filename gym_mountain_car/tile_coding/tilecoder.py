import gym
import numpy as np


class tilecoder:

    def __init__(self, env, numTilings, tiles1d):
        self.env = env
        self.maxIn = env.observation_space.high
        self.minIn = env.observation_space.low
        self.numTilings = numTilings
        # Define 1-D Size of tiling
        self.tiles1d = tiles1d
        self.dim = len(self.maxIn)
        # In this case with an tiles1d of 18 and numTilings of 4
        # each tile has 324 (18**dim) tiles with a total of 1296 tiles
        self.numTiles = (self.tiles1d ** self.dim) * self.numTilings
        self.actions = env.action_space.n
        # Combine these 1296 tiles with 3 possible actions
        # and we now have possible 3888 tiles
        self.n = self.numTiles * self.actions
        # Defines the physical size of the tile based on possible variables
        self.tileSize = np.divide(np.subtract(self.maxIn, self.minIn), self.tiles1d - 1)

    # Takes our current state and returns 4 integers / tile indices
    def getFeatures(self, variables):
        # Ensures lowest possible input is always 0
        self.variables = np.subtract(variables, self.minIn)
        tileIndices = np.zeros(self.numTilings)
        # Will take in state space and convert into tile indices
        matrix = np.zeros([self.numTilings, self.dim])
        for i in range(self.numTilings):
            for i2 in range(self.dim):
                matrix[i, i2] = int(self.variables[i2] / self.tileSize[i2] \
                                    + i / self.numTilings)
        for i in range(1, self.dim):
            matrix[:, i] *= self.tiles1d ** i
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tiles1d ** self.dim) + sum(matrix[i, :]))
        return tileIndices

    # Assigns actions values for all possible actions
    def getQ(self, features, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = self.getVal(theta, features, i)
        return Q

    # Calculates action values based upon theta
    def getVal(self, theta, features, action):
        val = 0
        for i in features:
            index = int(i + (self.numTiles * action))
            val += theta[index]
        return val

    # Creates a one hot vector for features so that theta can be updated
    def oneHotVector(self, features, action):
        oneHot = np.zeros(self.n)
        for i in features:
            index = int(i + (self.numTiles * action))
            oneHot[index] = 1
        return oneHot
