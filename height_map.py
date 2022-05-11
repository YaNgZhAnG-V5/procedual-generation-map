import numpy as np
from utils import *

class HeightMap:
    def __init__(self, mode):
        self.mode = mode

    def single_heightmap(self, mode):
        modefunc = getattr(self, mode + "_heightmap")
        modefunc()
        return self.elevation[:-1].copy()

    def mixed_heightmap(self):
        mode1, mode2 = self.mode.split("/")
        hm1 = self.single_heightmap(mode1)
        print("HM1:", mode1, hm1.max(), hm1.min(), hm1.mean())
        hm2 = self.single_heightmap(mode2)
        print("HM2:", mode2, hm2.max(), hm2.min(), hm2.mean())
        mix = 20 * (self.dvxs[:, 0] - self.dvxs[:, 1])
        mixing = 1 / (1 + np.exp(-mix))
        print("MIX:", mixing.max(), mixing.min(), mixing.mean())
        self.elevation[:-1] = mixing * hm2 + (1 - mixing) * hm1
        self.clean_coast()

    def shore_heightmap(self):
        print("Calculating elevations")
        n = self.nvxs
        self.elevation = np.zeros(n + 1)
        self.elevation[:-1] = 0.5 + ((self.dvxs - 0.5) * np.random.normal(0, 4, (1, 2))).sum(1)
        self.elevation[:-1] += -4 * (np.random.random() - 0.5) * distance(self.vxs, 0.5)
        mountains = np.random.random((50, 2))
        for m in mountains:
            self.elevation[:-1] += np.exp(-distance(self.vxs, m) ** 2 / (2 * 0.05 ** 2)) ** 2
        print("Edge height:", self.elevation[:-1][self.edge].max())

        along = (((self.dvxs - 0.5) * np.random.normal(0, 2, (1, 2))).sum(1) + np.random.normal(0, 0.5)) * 10
        self.erodability = np.exp(4 * np.arctan(along))

        for i in range(5):
            self.rift()
            self.relax()
        for i in range(5):
            self.relax()
        self.normalize_elevation()

        sealevel = np.random.randint(20, 40)
        self.raise_sealevel(sealevel)
        self.do_erosion(100, 0.025)

        self.raise_sealevel(np.random.randint(sealevel, sealevel + 20))
        self.clean_coast()

    def island_heightmap(self):
        self.erodability[:] = np.exp(
            np.random.normal(0, 4))
        theta = np.random.random() * 2 * np.pi
        dvxs = 0.5 * (self.vxs + self.dvxs)
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        y = (dvxs[:, 0] - .5) * -np.sin(theta) + (dvxs[:, 1] - .5) * np.cos(theta)
        self.elevation[:-1] = 0.001 * (1 - distance(self.vxs, 0.5))
        manhattan = np.max(np.abs(self.vxs - 0.5), 1)
        xs = x[(manhattan < 0.3) & (np.abs(y) < 0.1)]
        n = np.random.randint(2, 6)
        for i, u in enumerate(np.linspace(xs.min(), xs.max(), n) - 0.05):
            v = np.random.normal(0, 0.05)
            d = ((x - u) ** 2 + (y - v) ** 2) ** 0.5
            eruption = np.maximum(1 + 0.2 * i - d / 0.15, 0) ** 2
            print("Erupting", self.vxs[np.argmax(eruption), :])
            self.elevation[:-1] = np.maximum(
                self.elevation[:-1], eruption)
            self.do_erosion(20, 0.005)
            self.raise_sealevel(80)

        self.do_erosion(40, 0.005)

        maxheight = 30 * (0.46 - manhattan)
        self.elevation[:-1] = np.minimum(maxheight, self.elevation[:-1])

        sealevel = np.random.randint(85, 92)
        self.raise_sealevel(sealevel)
        self.clean_coast()

    def mountain_heightmap(self):
        theta = np.random.random() * 2 * np.pi
        dvxs = 0.5 * (self.vxs + self.dvxs)
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        y = (dvxs[:, 0] - .5) * -np.sin(theta) + (dvxs[:, 1] - .5) * np.cos(theta)
        self.elevation[:-1] = 50 - 10 * np.abs(x)
        mountains = np.random.random((50, 2))
        for m in mountains:
            self.elevation[:-1] += np.exp(-distance(self.vxs, m) ** 2 / (2 * 0.05 ** 2)) ** 2
        self.erodability[:] = np.exp(50 - 10 * self.elevation[:-1])
        for _ in range(5):
            self.rift()
        self.do_erosion(250, 0.02)
        self.elevation *= 0.5
        self.elevation[:-1] -= self.elevation[:-1].min() - 0.5

    def desert_heightmap(self):
        theta = np.random.random() * 2 * np.pi
        dvxs = self.dvxs
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        y = (dvxs[:, 0] - .5) * -np.sin(theta) + (dvxs[:, 1] - .5) * np.cos(theta)
        self.elevation[:-1] = x + 20 + 2 * self.perlin()
        self.erodability[:] = 1
        self.do_erosion(50, 0.005)
        self.elevation[:-1] -= self.elevation[:-1].min() - 0.1

    def clean_coast(self, n=3, outwards=True):
        for _ in range(n):
            new_elev = self.elevation[:-1].copy()
            for u in range(self.nvxs):
                if self.edge[u] or self.elevation[u] <= 0:
                    continue
                adjs = self.adj_vxs[u]
                adjelevs = self.elevation[adjs]
                if np.sum(adjelevs > 0) == 1:
                    new_elev[u] = np.mean(adjelevs[adjelevs <= 0])
            self.elevation[:-1] = new_elev
            if outwards:
                for u in range(self.nvxs):
                    if self.edge[u] or self.elevation[u] > 0:
                        continue
                    adjs = self.adj_vxs[u]
                    adjelevs = self.elevation[adjs]
                    if np.sum(adjelevs <= 0) == 1:
                        new_elev[u] = np.mean(adjelevs[adjelevs > 0])
                self.elevation[:-1] = new_elev

    def raise_sealevel(self, perc=35):
        maxheight = self.elevation.max()
        self.elevation -= np.percentile(self.elevation, perc)
        self.elevation *= maxheight / self.elevation.max()
        self.elevation[-1] = 0

    def do_erosion(self, n, rate=0.01):
        for _ in range(n):
            self.calc_downhill()
            self.calc_flow()
            self.calc_slopes()
            self.erode(rate)
            self.infill()
            self.elevation[-1] = 0