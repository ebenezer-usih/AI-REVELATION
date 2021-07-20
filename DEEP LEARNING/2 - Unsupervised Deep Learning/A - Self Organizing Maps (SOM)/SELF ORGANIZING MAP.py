#SELF ORGANIZING MAP

#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#TRAINING THE SOM
from minisom import MiniSom 
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration = 100)

#VISUALIZING THE RESULTS
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#FINDING THE FRAUDS
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(4, 8)], mappings[(5, 1)], mappings[(5, 2)], mappings[(6, 6)], mappings[(2, 2)], mappings[(3, 2)]), axis = 0)
frauds = sc.inverse_transform(frauds)