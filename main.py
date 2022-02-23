import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigh

from knn import decision_boundary_knn, assess_knn
from nn import decision_boundary_nn, assess_nn
from max_a_post import decision_boundary_map
from ged import decision_boundary_ged
from med import decision_boundary_med
from utils import gen_cluster, gen_ellipse, get_varname

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

global_colours = ['#000000', '#00ff00', '#ff0000']

# Output plot of points and contours, given list of class cluster coords and list of covariance matrices
def plot_points_contours(classes, means, covars, boundaries=None, file_name = "plot"):
  xs = []
  ys = []
  colours = []
  for i in range(len(classes)):

    class_label = 'Class ' + get_varname(means[i])[-1].upper() # colors & legend will correspond to class label
    
    xs += list(classes[i][0])
    ys += list(classes[i][1])
    colours += [class_label] * len(classes[i][0])  

    bound_xs = []
    bound_ys = []

    evals, evecs = eigh(covars[i])
    tempx, tempy = gen_ellipse(means[i], evecs[0], evecs[1], np.sqrt(evals[0]), np.sqrt(evals[1]))
    xs += list(tempx)
    ys += list(tempy)
    colours += [class_label] * len(list(tempx))

    if boundaries:
      b = []
      for b_type in boundaries:
        if b_type == 'nn':
          b.append(decision_boundary_nn(classes))
        elif b_type == 'knn':
          b.append(decision_boundary_knn(classes))
        elif b_type == 'map':
          b.append(decision_boundary_map(means, covars, classes))
        elif b_type == 'ged':
          b.append(decision_boundary_ged(means, covars, classes))
        elif b_type == 'med':
          b.append(decision_boundary_med(classes))

      for i in range(len(b)):
        for bound in b[i]:
          bound_xs.append([])
          bound_ys.append([])
          for point in bound:
            bound_xs[-1].append(point[0])
            bound_ys[-1].append(point[1])

  # -- using plotly --
  df = pd.DataFrame(data={'x': xs, 'y': ys, 'c': colours})
  fig = px.scatter(df, x='x', y='y', color='c')
  for i in range(len(bound_xs)):
    fig.add_trace(go.Scatter(x=bound_xs[i], y=bound_ys[i], mode='lines', line_color=global_colours[i]))  
  fig.update(layout_coloraxis_showscale=False)  # hide colorbar
  fig.update_layout(legend_title='Legend')
  fig.write_image(f"images/{file_name}.png")
  
  # -- using matplotlib --
  # plt.scatter(x=xs, y=ys, c=colours)
  # plt.savefig("plot.png")

if __name__ == '__main__':
  if not os.path.exists("images"):
    os.mkdir("images")
    
  x1 = gen_cluster(200, mean_a, var_a)
  x2 = gen_cluster(200, mean_b, var_b)
  # plot_points_contours([x1, x2], [mean_a, mean_b], [var_a, var_b], boundaries=['nn', 'knn'])
  plot_points_contours([x1, x2], [mean_a, mean_b], [var_a, var_b], boundaries=['ged'], file_name="ged")
  # plot_points_contours([x1, x2], [mean_a, mean_b], [var_a, var_b], boundaries=['med'], file_name="med")

  

  x3 = gen_cluster(100, mean_c, var_c)
  x4 = gen_cluster(200, mean_d, var_d)
  x5 = gen_cluster(150, mean_e, var_e)
  # plot_points_contours([x3, x4, x5], [mean_c, mean_d, mean_e], [var_c, var_d, var_e], boundaries=['nn', 'knn'], file_name="nn_knn_3_class")
  #plot_points_contours([x3, x4, x5], [mean_c, mean_d, mean_e], [var_c, var_d, var_e], boundaries=['map'], file_name="map_3_class")
  #plot_points_contours([x3, x4, x5], [mean_c, mean_d, mean_e], [var_c, var_d, var_e], boundaries=['ged'], file_name="ged_3_class")
  #plot_points_contours([x3, x4, x5], [mean_c, mean_d, mean_e], [var_c, var_d, var_e], boundaries=['med'], file_name="med_3_class")

  #err, cm = assess_knn([x1, x2], ['A', 'B'])
  #err, cm = assess_knn([x3, x4, x5], ['C', 'D', 'E'])
  #print(err)
  #print(cm)

