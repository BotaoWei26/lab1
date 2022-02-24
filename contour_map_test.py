import matplotlib.pyplot as plt
from utils import gen_cluster, ellipse_points
from classifiers import *
import numpy as np
from matplotlib.lines import Line2D


def get_feature_range(feats):
    xs, ys = np.array([]), np.array([])
    for feat in feats:
        xs = np.concatenate([xs, feat[0]])
        ys = np.concatenate([ys, feat[1]])
    return min(xs), min(ys), max(xs), max(ys)


def plot_points(ax, features):
    for feat in features:
        ax.scatter(*feat, s=2)

def plot_ellipse(ax, means, vars):
    colors = ["blue", "orange", "green"]
    for mean, var in zip(means, vars):
        x, y = ellipse_points(mean, var)
        ax.plot(x, y)

def boundary_contour(ax, classifier, feature_range, style, pad=2, delta=0.1):
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
        CS = ax.contour(X, Y, Zi, levels=[0.5], colors=colors[i], linestyles=style, linewidths=3-i*1)
        #ax.clabel(CS, inline=True, fontsize=10, fmt={0.5: "test"})


def make_graph(Ns, means, vars, xs, classifiers, out_file):
    fig, ax = plt.subplots(figsize=(10, 10))

    plot_points(ax, xs)
    plot_ellipse(ax, means, vars)

    feature_range = get_feature_range(xs)

    custom_legend = []
    custom_labels = []
    for classifier in classifiers:
        if classifier == "med":
            boundary_contour(ax, get_MED(means), feature_range, 'solid')
            custom_legend.append(Line2D([0], [0], color="black", linestyle="-"))
            custom_labels.append("MED")
        elif classifier == "ged":
            boundary_contour(ax, get_GED(means, vars), feature_range, 'dashed')
            custom_legend.append(Line2D([0], [0], color="black", linestyle="--"))
            custom_labels.append("GED")
        elif classifier == "map":
            boundary_contour(ax, get_MAP(Ns, means, vars), feature_range, 'dotted')
            custom_legend.append(Line2D([0], [0], color="black", linestyle=":"))
            custom_labels.append("MAP")
        elif classifier == "nn":
            boundary_contour(ax, get_kNN(xs, 1), feature_range, 'dashed', delta=0.5)
            custom_legend.append(Line2D([0], [0], color="black", linestyle="--"))
            custom_labels.append("NN")
        elif classifier == "knn":
            boundary_contour(ax, get_kNN(xs, 5), feature_range, 'dashed', delta=0.5)
            custom_legend.append(Line2D([0], [0], color="black", linestyle="--"))
            custom_labels.append("kNN k=5")

    plt.legend(custom_legend, custom_labels)
    plt.savefig(out_file)
    print(f"{out_file} created")


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


make_graph(
    (Na, Nb),
    (mean_a, mean_b),
    (var_a, var_b),
    (xa, xb),
    classifiers=("med", "ged", "map"),
    out_file="images/case1_med_ged_map.png"
)

make_graph(
    (Na, Nb),
    (mean_a, mean_b),
    (var_a, var_b),
    (xa, xb),
    classifiers=("nn",),
    out_file="images/case1_nn.png"
)

make_graph(
    (Na, Nb),
    (mean_a, mean_b),
    (var_a, var_b),
    (xa, xb),
    classifiers=("knn",),
    out_file="images/case1_knn.png"
)

make_graph(
    (Nc, Nd, Ne),
    (mean_c, mean_d, mean_e),
    (var_c, var_d, var_e),
    (xc, xd, xe),
    classifiers=("med", "ged", "map"),
    out_file="images/case2_med_ged_map.png"
)

make_graph(
    (Nc, Nd, Ne),
    (mean_c, mean_d, mean_e),
    (var_c, var_d, var_e),
    (xc, xd, xe),
    classifiers=("nn",),
    out_file="images/case2_nn.png"
)

make_graph(
    (Nc, Nd, Ne),
    (mean_c, mean_d, mean_e),
    (var_c, var_d, var_e),
    (xc, xd, xe),
    classifiers=("knn",),
    out_file="images/case2_knn.png"
)