import numpy as np
from collections import defaultdict
import pickle
import matplotlib as mpl
import gzip


def distance(a, b):
    disp2 = (a - b) ** 2
    if len(disp2.shape) == 2:
        return np.sum(disp2, 1) ** 0.5
    else:
        return np.sum(disp2) ** 0.5


def trislope(xys, zs):
    a = np.matrix(np.column_stack((xys, [1, 1, 1])))
    b = np.asarray(zs).reshape((3, 1))
    x = a.I * b
    return x[0, 0], x[1, 0]


def relaxpts(pts, idxs, n=1):
    adj = defaultdict(list)
    for p1, p2 in idxs:
        adj[p1].append(p2)
        adj[p2].append(p1)
    for _ in range(n):
        newpts = pts.copy()
        for p in adj:
            if len(adj[p]) == 1:
                continue
            adjs = adj[p] + [p]
            newpts[p, :] = np.mean(pts[adjs, :], 0)
        pts = newpts
    return [[(pts[p1, 0], pts[p1, 1]), (pts[p2, 0], pts[p2, 1])]
            for p1, p2 in idxs]


def mergelines(segs):
    n = len(segs)
    segs = set((tuple(a), tuple(b)) for (a, b) in segs)
    assert len(segs) == n
    adjs = defaultdict(list)
    for a, b in segs:
        adjs[a].append((a, b))
        adjs[b].append((a, b))
    lines = []
    line = None
    nremoved = 0
    length = 0
    while segs:
        if line is None:
            line = list(segs.pop())
            nremoved += 1
        found = None
        for seg in adjs[line[-1]]:
            if seg not in segs:
                continue
            if seg[0] == line[-1]:
                line.append(seg[1])
                found = seg
                break
            elif seg[1] == line[-1]:
                line.append(seg[0])
                found = seg
                break
        if found:
            segs.remove(found)
            nremoved += 1
            continue
        for seg in adjs[line[0]]:
            if seg not in segs:
                continue
            if seg[0] == line[0]:
                line.insert(0, seg[1])
                found = seg
                break
            elif seg[1] == line[0]:
                line.insert(0, seg[0])
                found = seg
                break
        if found:
            segs.remove(found)
            nremoved += 1
            continue
        # nothing found

        lines.append(mpl.path.Path(line))
        length += len(line) - 1
        line = None
    assert nremoved == n, "Got %d, Removed %d" % (n, nremoved)
    if line is not None:
        length += len(line) - 1
        lines.append(mpl.path.Path(line))
    print(length)
    return lines


def load(filename):
    with gzip.open(filename) as f:
        return pickle.loads(f.read())


city_counts = {
    'shore': (12, 20),
    'island': (5, 10),
    'mountain': (10, 25),
    'desert': (5, 10)
}
terr_counts = {
    'shore': (3, 7),
    'island': (2, 4),
    'mountain': (3, 6),
    'desert': (3, 5)
}
riverpercs = {
    'shore': 5,
    'island': 3,
    'mountain': 8,
    'desert': 1
}