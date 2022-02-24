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

def ellipse_points(mu, sigma, n_pts=500):
    """
    Generates 2D points on the unit standard deviation contour
    of N(mu, sigma). Points will be evenly distributed (radially)
    and the first and last point will be the same.

    Parameters
    ----------
    mu : array-like
        The mean of the Gaussian distribution.
    sigma : array-like
        The covariance matrix of the Gaussian distribution.
    n_pts : int, optional
        The number of points to generate.

    Returns
    -------
    x1 : ndarray
        The x1 coordinates of the generated points.
    x2 : ndarray
        The x2 coordinates of the generated points.

    """
    sigma = np.asarray(sigma, dtype=np.float32)
    if len(mu) != 2 or sigma.shape != (2, 2):
      raise ValueError(
        'mu and sigma must be valid for a 2D Gaussian distribution')

    # Get eigenvalues
    eigvals = np.linalg.eigvals(sigma)
    lambda1 = eigvals.max()
    lambda2 = eigvals.min()

    # Get angle between ellipse and x1-axis
    theta = _ellipse_angle(sigma, lambda1=lambda1)

    # Initialize parameter t
    t = np.linspace(0, 2 * np.pi, n_pts)

    # Generate points
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x1 = np.sqrt(lambda1) * np.cos(theta) * cos_t - np.sqrt(lambda2) * np.sin(theta) * sin_t + mu[0]
    x2 = np.sqrt(lambda1) * np.sin(theta) * cos_t + np.sqrt(lambda2) * np.cos(theta) * sin_t + mu[1]

    return x1, x2


def _ellipse_angle(sigma, lambda1=None):
  """
  Computes the angle between the semi-major axis of sigma's
  unit standard deviation contour and the horizontal (x1) axis.

  Parameters
  ----------
  sigma : array-like
      The covariance matrix of the Gaussian distribution.
  lambda1 : float, optional
      The largest eigenvalue of sigma. If not given, it will
      be computed from sigma.

  Returns
  -------
  theta : float
      The angle between the semi-major axis of sigma's unit
      standard deviation contour and the horizontal (x1) axis.

  """
  if not sigma[0, 1]:
    # Axis-aligned case
    theta = 0 if sigma[0, 0] >= sigma[1, 1] else np.pi / 2
  else:
    # Oblique case
    lambda1 = np.linalg.eigvals(sigma).max() if lambda1 is None else lambda1
    theta = np.arctan2(lambda1 - sigma[0, 0], sigma[0, 1])
  return theta
