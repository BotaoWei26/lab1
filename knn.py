import numpy as np
from utils import euclidean, gen_cluster

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

k = 5
boundary_range_x = [0, 20] # range for which decision boundary is found
boundary_range_y = [0, 20] # range for which decision boundary is found
boundary_precision = 0.1 # precision of grid used for boundary search
boundary_threshold = 0.07 # acceptable threshold to be considered part of boundary

# Helper - returns (x, y), distance for mean of k nearest neighbours
def center_knn(c, point, k):
  dists = []
  nns = []
  for i in range(len(c[0])):
    curr = (c[0][i], c[1][i])
    dists.append(euclidean(point, curr))
  for i in range(k):
    min_pos = dists.index(min(dists))
    nns.append((c[0][min_pos], c[1][min_pos]))
    dists[min_pos] = 999999
  mean_x, mean_y = 0, 0
  for p in nns:
    mean_x += p[0]
    mean_y += p[1]
  mean_x /= k
  mean_y /= k
  return (mean_x, mean_y), euclidean(point, (mean_x, mean_y))

def decision_boundary_knn(classes):

  pairs = []
  boundary_points = []
  if len(classes) == 3:
    pairs = [(0, 1), (0, 2), (1, 2)]
    boundary_points = [[], [], []]
  else:
    pairs = [(0, 1)]
    boundary_points = [[]]

  for x1 in range(int(boundary_range_x[0]/boundary_precision), int(boundary_range_x[1]/boundary_precision)):
    for y1 in range(int(boundary_range_y[0]/boundary_precision), int(boundary_range_y[1]/boundary_precision)):
      x = x1 * boundary_precision
      y = y1 * boundary_precision
      nn_list = [] # each element is (x, y), euclidean distance
      for c in classes:
        nn_list.append(center_knn(c, (x, y), k))
      for i in range(len(pairs)):
        diff = abs(nn_list[pairs[i][1]][1] - nn_list[pairs[i][0]][1])
        if diff < boundary_threshold:
          boundary_points[i].append((x, y))

  return boundary_points

def classify_point_knn(classes, point):
  centers = []
  for c in classes:
    centers.append(center_knn(c, point, k))
  min_dist = 999999
  predicted = 0
  for i in range(len(centers)):
    if centers[i][1] < min_dist:
      predicted = i
      min_dist = centers[i][1]
  return predicted
