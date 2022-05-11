import numpy as np
from utils import *
import heapq


class CitiesPlacement:
    def __init__(self, flow, elevation, vxs, adj_vxs, edge_weight):
        self.flow = flow
        self.elevation = elevation
        self.cities = []
        self.vxs = vxs
        self.nvxs = self.vxs.shape[0]
        self.adj_vxs = adj_vxs
        self.edge_weight = edge_weight

    def place_cities(self, n=20):
        city_score = self.flow ** 0.5
        city_score[self.elevation[:-1] <= 0] = -9999999
        self.cities = []
        while len(self.cities) < n:
            newcity = np.argmax(city_score)
            if np.random.random() < (len(self.cities) + 1) ** -0.2 and \
                    0.1 < self.vxs[newcity, 0] < 0.9 and \
                    0.1 < self.vxs[newcity, 1] < 0.9:
                self.cities.append(newcity)
            city_score -= 0.01 * 1 / (distance(self.vxs, self.vxs[newcity, :]) + 1e-9)
        return self.cities

    def grow_territory(self, n=7):
        done = np.zeros(self.nvxs, np.int32) - 1
        q = []
        for city in self.cities[:n]:
            heapq.heappush(q, (0, city, city))
        while q:
            dist, vx, city = heapq.heappop(q)
            if done[vx] != -1:
                continue
            done[vx] = city
            for u in self.adj_vxs[vx]:
                if done[u] != -1 or u == -1:
                    continue
                newdist = self.edge_weight(u, vx, territory=True)
                heapq.heappush(q, (dist + newdist, u, city))
        territories = done
        return territories
