import numpy as np

boundary_range_x = [0, 20] # range for which decision boundary is found
boundary_range_y = [0, 20] # range for which decision boundary is found
boundary_precision = 0.1 # precision of grid used for boundary search
boundary_threshold = 0.07 # acceptable threshold to be considered part of boundary

def bivariate_gaussian_prob(x, mean, var):
  '''  
  finds the bivariate, conditional/mariginal Gaussian probability for some class A
  
  x: random column vector of dimension n
  mean: 1xn vector
  var: nxn covariance matrix
  
  returns: the conditional probability p(x|A) for an n=2 Gaussian distribution
  '''
  scalar_term = 1/(2 * np.pi * np.linalg.det(var) ** 0.5)
  expon_term = -0.5 * (x-mean).dot(np.linalg.inv(var)).dot(x-mean)
  return scalar_term * np.exp(expon_term)


def decision_boundary_map(means, vars, classes):
  '''
  creates the decision boundary for MAP classifier. 
  params follow the following form: [class1, class2, class3...] etc.

  means: means for all class
  vars: variances for all classes
  classes: vector of vectors. inner vecs are data for each class in the form (x,y).
  '''
  # priors are proportional to the number of samples in each class
  total_samples = sum([len(x) for x in classes])
  priors = [len(x)/total_samples for x in classes] # p(A), p(B) etc.

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
        likelihood_ratio = bivariate_gaussian_prob(point, means[a], vars[a])/bivariate_gaussian_prob(point, means[b], vars[b])
        prior_ratio = priors[a]/priors[b]
        
        diff = abs(likelihood_ratio - prior_ratio)
        if diff < boundary_threshold:
          boundary_points[i].append((x, y))

  return boundary_points