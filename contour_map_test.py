import numpy as np
import matplotlib.pyplot as plt
from utils import gen_cluster, gen_ellipse
from classifiers import *


def get_feature_range(feats):
    xs, ys = np.array([]), np.array([])
    for feat in feats:
        xs = np.concatenate([xs, feat[0]])
        ys = np.concatenate([ys, feat[1]])
    return min(xs), min(ys), max(xs), max(ys)


def plot_points(ax, features):
    for feat in features:
        ax.scatter(*feat, s=2)


def boundary_contour(ax, classifier, feature_range, style, pad=1, delta=0.1):
    minx, miny, maxx, maxy = feature_range
    xs = np.arange(minx-pad, maxx+pad, delta)
    ys = np.arange(miny-pad, maxy+pad, delta)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros((len(ys), len(xs)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[j,i] = classifier((x,y))

    colors = ["blue", "orange", "green"]
    for i in range(int(np.max(Z)) + 1):
        Zi = Z == i
        CS = ax.contour(X, Y, Zi, levels=[0.5], colors=colors[i], linestyles=style, linewidths=3-i*0.5)
    #ax.clabel(CS, inline=True, fontsize=10)



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

Na = 200
Nb = 200
Nc = 100
Nd = 200
Ne = 150

xa = gen_cluster(Na, mean_a, var_a)
xb = gen_cluster(Nb, mean_b, var_b)

xc = gen_cluster(Nc, mean_c, var_c)
xd = gen_cluster(Nd, mean_d, var_d)
xe = gen_cluster(Ne, mean_e, var_e)


fig, ax1 = plt.subplots()

plot_points(ax1, (xa, xb))

feature_range = get_feature_range((xa, xb))
med_classifier1 = get_MED((mean_a, mean_b))
boundary_contour(ax1, med_classifier1, feature_range, 'solid')

ged_classifier1 = get_GED((mean_a, mean_b), (var_a, var_b))
boundary_contour(ax1, ged_classifier1, feature_range, 'dashed')

map_classifier1 = get_MAP((mean_a, mean_b), (var_a, var_b), (Na, Nb))
boundary_contour(ax1, map_classifier1, feature_range, 'dotted')

plt.savefig(f"images/1.png")

fig, ax2 = plt.subplots()

plot_points(ax2, (xc, xd, xe))

feature_range = get_feature_range((xc, xd, xe))
med_classifier2 = get_MED((mean_c, mean_d, mean_e))
boundary_contour(ax2, med_classifier2, feature_range, 'solid')

ged_classifier2 = get_GED((mean_c, mean_d, mean_e), (var_c, var_d, var_e))
boundary_contour(ax2, ged_classifier2, feature_range, 'dashed')

map_classifier2 = get_MAP((mean_c, mean_d, mean_e), (var_c, var_d, var_e), (Nc, Nd, Ne))
boundary_contour(ax2, map_classifier2, feature_range, 'dotted')

plt.savefig(f"images/2.png")

