import numpy as np
from height_map import HeightMap
from map_generation.utils import *


class DesertHeightMap(HeightMap):
    def get_heightmap(self):
        theta = np.random.random() * 2 * np.pi
        dvxs = self.dvxs
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        y = (dvxs[:, 0] - .5) * -np.sin(theta) + (dvxs[:, 1] - .5) * np.cos(theta)
        self.elevation[:-1] = x + 20 + 2 * perlin(self.vxs)
        self.erodability[:] = 1
        self.do_erosion(50, 0.005)
        self.elevation[:-1] -= self.elevation[:-1].min() - 0.1
        self.finalize()
