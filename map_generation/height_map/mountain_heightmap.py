import numpy as np
from map_generation.height_map.height_map import HeightMap
from map_generation.utils import *


class MountainHeightMap(HeightMap):
    def get_heightmap(self):
        theta = np.random.random() * 2 * np.pi
        dvxs = 0.5 * (self.vxs + self.dvxs)

        # draw a centered mountain ridge
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        self.elevation[:-1] = 50 - 10 * np.abs(x)

        # create other mountains
        mountains = np.random.random((50, 2))
        for m in mountains:
            self.elevation[:-1] += np.exp(-distance(self.vxs, m) ** 2 / (2 * 0.05 ** 2)) ** 2
        self.erodability[:] = np.exp(50 - 10 * self.elevation[:-1])
        for _ in range(5):
            self.rift()
        self.do_erosion(250, 0.02)
        self.elevation *= 0.5
        self.elevation[:-1] -= self.elevation[:-1].min() - 0.5
        self.finalize()
        return self.downhill, self.flow, self.slope
