import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm
from math import pi
import inspect

num_points_ellipse = 2000 # number of points generated for equiprobability contours

# Helper - computes euclidean distance between two points
def euclidean(x1, x2):
  return np.sqrt(((x2[1] - x1[1])**2) + ((x2[0] - x1[0])**2))

# Generate cluster of data points from distribution
# Assumes 2 variables involved (mean is 1x2, var is 2x2)
def gen_cluster(n, mean, var):
  x = norm.rvs(size=(2, n))
  c = cholesky(var, lower=True)
  shifted_x = np.dot(c, x)
  shifted_x[0] = shifted_x[0] + mean[0]
  shifted_x[1] = shifted_x[1] + mean[1]
  return shifted_x

# ax1, ax2 are axis vectors - eigenvectors of covariance
# a, b are axis lengths - roots of eigenvalues of covariance
def gen_ellipse(c, ax1, ax2, a, b):
  r = np.array([ax1, ax2]).T
  t = np.linspace(0, 2*pi, num_points_ellipse)
  x = a*np.cos(t)
  y = b*np.sin(t)
  x, y = np.dot(r, [x, y])
  x += c[0]
  y += c[1]
  return x, y


# get variable name
def get_varname(var):
  callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
  name = [var_name for var_name, var_val in callers_local_vars if var_val is var]
  return name[0]