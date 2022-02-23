import numpy as np

boundary_range_x = [0, 20] # range for which decision boundary is found
boundary_range_y = [0, 20] # range for which decision boundary is found
boundary_precision = 0.1 # precision of grid used for boundary search
boundary_threshold = 0.01 # acceptable threshold to be considered part of boundary

def ged(x, z, var):
  '''  
  x: random column vector of dimension n
  z: mean of class
  var: nxn covariance matrix
  
  returns: the ged metric for a given class
  '''
  return ((x-z).dot(np.linalg.inv(var)).dot(x-z)) ** 0.5

def decision_boundary_ged(means, vars, classes):
  pairs = []
  boundary_points = []
  if len(classes) == 3:
    pairs = [(0, 1), (0, 2), (1, 2)]
    boundary_points = [[], [], []] # between classes 0 and 1, 0 and 2, 1 and 2
  else:
    pairs = [(0, 1)]
    boundary_points = [[]] # between classes 0 and 1

  for x in np.arange(boundary_range_x[0], boundary_range_x[1], boundary_precision):
    for y in np.arange(boundary_range_y[0], boundary_range_y[1], boundary_precision):
      point = np.array([x,y])
      for i in range(len(pairs)):
        a, b = pairs[i]
        
        diff = abs(ged(point, means[a], vars[a]) - ged(point, means[b], vars[b]))
        if diff < boundary_threshold:
          boundary_points[i].append((x, y))

  return boundary_points