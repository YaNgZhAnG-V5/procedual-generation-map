# coding: utf-8
import random

from height_map import ShoreHeightMap, IslandHeightMap, MountainHeightMap, DesertHeightMap
from language import Language
from civilization import CitiesPlacement
from plotter import Plotter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.spatial as spl
import scipy.sparse as spa
import scipy.sparse.csgraph as csg
import heapq
import noise
import itertools
from utils import *

plt.rcParams['font.family'] = "Palatino Linotype"
plt.rcParams['font.size'] = 10


class MapGrid(object):
    def __init__(self, mode='shore', n=16384):
        self.pts = np.random.random((n, 2))
        self.improve_pts()
        self.vor = spl.Voronoi(self.pts)
        self.regions = [self.vor.regions[i] for i in self.vor.point_region]
        self.vxs = self.vor.vertices
        self.build_adjs()
        self.improve_vxs()
        self.calc_edges()
        self.distort_vxs()
        self.elevation = np.zeros(self.vxs.shape[0] + 1)
        self.erodability = np.ones(self.vxs.shape[0])

        self.mode = mode
        if '/' in mode:
            self.mixed_heightmap()
            mode = mode.split("/")[0]
        else:
            self.single_heightmap(mode)
        self.riverperc = riverpercs[mode] * np.mean(self.elevation > 0)

        cities_placer = CitiesPlacement(self.flow, self.elevation, self.vxs, self.adj_vxs, self.edge_weight)
        self.cities = cities_placer.place_cities(np.random.randint(*city_counts[mode]))
        self.territories = cities_placer.grow_territory(np.random.randint(*terr_counts[mode]))
        self.lang = Language()
        self.city_names, self.region_names = self.lang.name_places(self.cities, self.territories)

        self.path_cache = {}
        self.fill_path_cache(self.big_cities)

    @property
    def big_cities(self):
        return [c for c in self.cities if self.territories[c] == c]

    def save(self, filename):
        with gzip.open(filename, "w") as f:
            f.write(pickle.dumps(self))

    def improve_pts(self, n=2):
        print("Improving points")
        for _ in range(n):
            vor = spl.Voronoi(self.pts)
            newpts = []
            for idx in range(len(vor.points)):
                pt = vor.points[idx, :]
                region = vor.regions[vor.point_region[idx]]
                if -1 in region:
                    newpts.append(pt)
                else:
                    vxs = np.asarray([vor.vertices[i, :] for i in region])
                    vxs[vxs < 0] = 0
                    vxs[vxs > 1] = 1
                    newpt = np.mean(vxs, 0)
                    newpts.append(newpt)
            self.pts = np.asarray(newpts)

    def improve_vxs(self):
        print("Improving vertices")
        n = self.vxs.shape[0]
        for v in range(n):
            self.vxs[v, :] = np.mean(self.pts[self.vx_regions[v]], 0)

    def build_adjs(self):
        print("Building adjacencies")
        self.adj_pts = defaultdict(list)
        self.adj_vxs = defaultdict(list)
        for p1, p2 in self.vor.ridge_points:
            self.adj_pts[p1].append(p2)
            self.adj_pts[p2].append(p1)
        for v1, v2 in self.vor.ridge_vertices:
            self.adj_vxs[v1].append(v2)
            self.adj_vxs[v2].append(v1)
        self.vx_regions = defaultdict(list)
        for p in range(self.pts.shape[0]):
            for v in self.regions[p]:
                if v != -1:
                    self.vx_regions[v].append(p)
        self.tris = defaultdict(list)
        for p in range(self.pts.shape[0]):
            for v in self.regions[p]:
                self.tris[v].append(p)
        self.adj_mat = np.zeros((self.vxs.shape[0], 3), np.int32) - 1
        for k, v in self.adj_vxs.items():
            if k != -1:
                self.adj_mat[k, :] = v

    def calc_edges(self):
        n = self.vxs.shape[0]
        self.edge = np.zeros(n, bool)
        for u in range(n):
            adjs = self.adj_vxs[u]
            if -1 in adjs:  # or \
                #                     np.all(self.vxs[adjs,0] > self.vxs[u,0]) or \
                #                     np.all(self.vxs[adjs,0] < self.vxs[u,0]) or \
                #                     np.all(self.vxs[adjs,1] > self.vxs[u,1]) or \
                #                     np.all(self.vxs[adjs,1] < self.vxs[u,1]):
                self.edge[u] = True

    def distort_vxs(self):
        self.dvxs = self.vxs.copy()
        self.dvxs[:, 0] += perlin(self.vxs)
        self.dvxs[:, 1] += perlin(self.vxs)

    def single_heightmap(self, mode):
        # modefunc = getattr(self, mode + "_heightmap")
        # modefunc()
        if mode == "shore":
            height_map_generator = ShoreHeightMap(self)
        elif mode == "island":
            height_map_generator = IslandHeightMap(self)
        elif mode == "mountain":
            height_map_generator = MountainHeightMap(self)
        elif mode == "desert":
            height_map_generator = DesertHeightMap(self)
        else:
            raise NotImplementedError
        self.downhill, self.flow, self.slope = height_map_generator.get_heightmap()
        self.set_values_from_height_map_generator(height_map_generator)
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

    def set_values_from_height_map_generator(self, height_map):
        self.erodability = height_map.erodability
        self.elevation = height_map.elevation
        self.vxs = height_map.vxs
        self.dvxs = height_map.dvxs
        self.adj_vxs = height_map.adj_vxs
        self.adj_mat = height_map.adj_mat
        self.edge = height_map.edge
        self.regions = height_map.regions
        self.pts = height_map.pts
        self.elevation_pts = height_map.elevation_pts

    def clean_coast(self, n=3, outwards=True):
        for _ in range(n):
            new_elev = self.elevation[:-1].copy()
            for u in range(self.vxs.shape[0]):
                if self.edge[u] or self.elevation[u] <= 0:
                    continue
                adjs = self.adj_vxs[u]
                adjelevs = self.elevation[adjs]
                if np.sum(adjelevs > 0) == 1:
                    new_elev[u] = np.mean(adjelevs[adjelevs <= 0])
            self.elevation[:-1] = new_elev
            if outwards:
                for u in range(self.vxs.shape[0]):
                    if self.edge[u] or self.elevation[u] > 0:
                        continue
                    adjs = self.adj_vxs[u]
                    adjelevs = self.elevation[adjs]
                    if np.sum(adjelevs <= 0) == 1:
                        new_elev[u] = np.mean(adjelevs[adjelevs > 0])
                self.elevation[:-1] = new_elev

    def edge_weight(self, u, v, territory=False):
        horiz = distance(self.vxs[u, :], self.vxs[v, :])
        vert = self.elevation[v] - self.elevation[u]
        if vert < 0:
            vert /= 10
        difficulty = 1 + (vert / horiz) ** 2
        if territory:
            difficulty += 100 * self.flow[u] ** 0.5
        else:
            if self.downhill[u] == v or self.downhill[v] == u:
                difficulty *= 0.9
        if self.elevation[u] <= 0:
            difficulty = 1
        if (self.elevation[u] <= 0) != (self.elevation[v] <= 0):
            difficulty = 2000 if territory else 200
        return horiz * difficulty

    def fill_path_cache(self, cities):
        cities = list(cities)
        n = self.vxs.shape[0]
        g = spa.dok_matrix((n + 2, n + 2))
        edge = self.extend_area(self.edge, 5)
        for u in range(n):
            for v in self.adj_vxs[u]:
                if not (edge[u] and edge[v]):
                    g[u, v] = self.edge_weight(u, v)
            if edge[u]:
                d = self.vxs[u, 0] - self.vxs[u, 1]
                if d < -0.5:
                    g[n, u] = 1e-12
                if d > 0.5:
                    g[u, n + 1] = 1e-12
        g = g.tocsc()
        tocities = cities + [n + 1]
        fromcities = cities + [n]
        dists, preds = csg.dijkstra(g,
                                    indices=fromcities,
                                    return_predecessors=True)

        for b in tocities:
            for i, a in enumerate(fromcities):
                if a == b: continue
                p = [b]
                while p[0] != a:
                    p.insert(0, preds[i, p[0]])
                p = [x for x in p if x < n]
                d = dists[i, b]
                self.path_cache['topleft' if a == n else a,
                                'bottomright' if b == n + 1 else b] = p, d

    def shortest_path(self, start, end, ):
        try:
            return self.path_cache[start, end]
        except KeyError:
            print("WARNING: Uncached path search", start, end)
            pass
        best_dir = {}
        q = []
        flipped = False
        if isinstance(end, str):
            flipped = True
            start, end = end, start
        heapq.heappush(q, (0, 0, end, -1))
        while start not in best_dir:
            _, dist, u, v = heapq.heappop(q)
            if u in best_dir:
                continue
            best_dir[u] = v
            if self.territories[u] == u:
                path = [u]
                while path[-1] != end:
                    path.append(best_dir[path[-1]])
                if flipped:
                    self.path_cache[end, u] = path[::-1], dist
                    print("CACHE", len(self.path_cache))
                else:
                    self.path_cache[u, end] = path, dist
                    print("CACHE", len(self.path_cache))
            length = dist
            if isinstance(start, str) and self.edge[u]:
                if (start == "topleft" and
                    self.vxs[u, 0] - self.vxs[u, 1] < -0.5) or \
                        (start == "bottomright" and
                         self.vxs[u, 0] - self.vxs[u, 1] > 0.5):
                    start = u
                    break
            for w in self.adj_vxs[u]:
                if w == -1 or w in best_dir or (self.edge[u] and self.edge[w]):
                    continue
                est = distance(self.vxs[end, :], self.vxs[w, :]) * 0.85
                d = dist + self.edge_weight(w, u)
                heapq.heappush(q, (d + est, d, w, u))
        path = [start]
        while path[-1] != end:
            path.append(best_dir[path[-1]])
        if flipped:
            self.path_cache[end, start] = path[::-1], length
            print("CACHE", len(self.path_cache))
            return path[::-1], length
        else:
            self.path_cache[start, end] = path, length
            print("CACHE", len(self.path_cache))
            return path, length

    def extend_area(self, area, n):
        for _ in range(10):
            adj = self.adj_mat[area, :]
            area[adj[adj != -1]] = True
        return area

    def ordered_cities(self):
        cities = self.big_cities
        dists = {}
        for c in cities:
            for d in cities:
                if c == d:
                    continue
                dists[c, d] = self.shortest_path(c, d)[1]
            dists["topleft", c] = self.shortest_path("topleft", c)[1]
            dists[c, "bottomright"] = self.shortest_path(c, "bottomright")[1]

        def totallength(seq):
            seq = ["topleft"] + list(seq) + ["bottomright"]
            return sum(dists[c, d] for c, d in zip(seq[:-1], seq[1:]))

        clist = min(itertools.permutations(cities),
                    key=totallength)
        return clist


if __name__ == '__main__':
    seed = 2000
    random.seed(seed)
    np.random.seed(seed)
    for i in range(1):
        for mode in ["shore", "island", "mountain", "desert"]:

            plt.close('all')
            while True:
                m = MapGrid(mode=mode)
                filename = "../tests/%s-%02d.png" % (m.mode, i)
                plotter = Plotter(m)
                plotter.plot(filename)
                break

