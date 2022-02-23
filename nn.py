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

boundary_range_x = [0, 20] # range for which decision boundary is found
boundary_range_y = [0, 20] # range for which decision boundary is found
boundary_precision = 0.1 # precision of grid used for boundary search
boundary_threshold = 0.07 # acceptable threshold to be considered part of boundary

# Helper - returns (x, y), distance for nearest neighbour
def center_nn(c, point):
  '''best_dist = 999999
  nn = None
  for i in range(len(c[0])):
    curr = (c[0][i], c[1][i])
    curr_dist = euclidean(point, curr)
    if curr_dist < best_dist:
      best_dist = curr_dist
      nn = curr'''
  dists = []
  for i in range(len(c[0])):
    curr = (c[0][i], c[1][i])
    dists.append(euclidean(point, curr))
  best_dist = min(dists)
  min_pos = dists.index(best_dist)
  nn = (c[0][min_pos], c[1][min_pos])
  return nn, best_dist

def decision_boundary_nn(classes):

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
        nn_list.append(center_nn(c, (x, y)))
      for i in range(len(pairs)):
        diff = abs(nn_list[pairs[i][1]][1] - nn_list[pairs[i][0]][1])
        if diff < boundary_threshold:
          boundary_points[i].append((x, y))

  return boundary_points

def classify_point_nn(classes, point):
  centers = []
  for c in classes:
    centers.append(center_nn(c, point))
  min_dist = 999999
  predicted = 0
  for i in range(len(centers)):
    if centers[i][1] < min_dist:
      predicted = i
      min_dist = centers[i][1]
  return predicted

# Assess experimental error - classes is the 'train set' (data used for the classifier)
# Also return confusion matrices
def assess_nn(classes, class_names):

  # Generate test set for each class
  test = []
  mean = None
  covar = None
  n = 0
  
  for c in class_names:
    if c == 'A':
      mean = mean_a
      covar = var_a
      n = 200
    elif c == 'B':
      mean = mean_b
      covar = var_b
      n = 200
    elif c == 'C':
      mean = mean_c
      covar = var_c
      n = 100
    elif c == 'D':
      mean = mean_d
      covar = var_d
      n = 200
    elif c == 'E':
      mean = mean_e
      covar = var_e
      n = 150
    test.append(gen_cluster(n, mean, covar))

  # Classify each point in test set
  dim = len(test)
  cm = np.zeros((dim, dim))

  # Tally confusion matrix
  # True class is i - rows (first dim of cm)
  for i in range(len(test)):
    for j in range(len(test[i][0])):
      point = (test[i][0][j], test[i][1][j])
      predicted = classify_point_nn(classes, point) # columns (second dim of cm)
      cm[i][predicted] += 1

  # Experimental errors
  if len(cm) == 3:
    correct = [0, 0, 0]
    incorrect = [0, 0, 0]
  else:
    correct = [0, 0]
    incorrect = [0, 0]
  for i in range(len(cm)):
    for j in range(len(cm)):
      if i == j:
        correct[i] += cm[i][j]
      else:
        incorrect[i] += cm[i][j]

  err = [incorrect[i] / (correct[i] + incorrect[i]) for i in range(len(correct))]

  return err, cm