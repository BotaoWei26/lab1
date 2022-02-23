import numpy as np
import itertools
from utils import gen_cluster, euclidean

num_boundary_points = 100

mean_a = np.array([5, 10])
mean_b = np.array([10, 15])
mean_c = np.array([5, 10])
mean_d = np.array([15, 10])
mean_e = np.array([10, 5])

var_a = np.array([[8, 0], [0, 4]])
var_b = np.array([[8, 0], [0, 4]])
var_c = np.array([[8, 4], [4, 40]])
var_d = np.array([[8, 0], [0, 8]])
var_e = np.array([[10, -5], [-5, 20]])


def decision_boundary_med(cls):
    lines = []
    for pair in itertools.combinations(cls, 2):
        cl1 = np.array(pair[0])
        cl2 = np.array(pair[1])
        lines.append(med(cl1, cl2))

    return lines


def sample_mean(feat):
    feat = np.array(feat).T
    n, d = feat.shape
    z = np.zeros(d)
    for sample in feat:
        z += sample

    z /= n
    return z


def boundaries(cl1, cl2):
    all_x = np.concatenate([cl1[0], cl2[0]])
    all_y = np.concatenate([cl1[1], cl2[1]])
    return min(all_x), min(all_y), max(all_x), max(all_y)


def med(cl1, cl2):
    minx, miny, maxx, maxy = boundaries(cl1, cl2)
    z1, z2 = sample_mean(cl1), sample_mean(cl2)

    w1, w2 = z1 - z2
    w0 = (1/2) * (np.dot(z2, z2) - np.dot(z1, z1))
    if w2 != 0:
        xs = np.linspace(minx, maxx, num_boundary_points)
        points = np.array([(x, (w0 + w1*x)/-w2) for x in xs])
        boundary_points = points[((points[:,1] >= miny) & (points[:,1] <= maxy))]
    elif w1 != 0:
        ys = np.arange(miny, maxy, num_boundary_points)
        boundary_points = [(-w0 / w1, y) for y in ys]
    else:
        boundary_points = []
    return boundary_points




