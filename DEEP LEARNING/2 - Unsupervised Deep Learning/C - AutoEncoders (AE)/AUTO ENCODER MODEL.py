#AUTOENCODERS

#IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#IMPORTING THE DATASETS
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') #engine is used for proper importation
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#PREPARING THE TRAINING SET AND THE TEST SET
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t', header = None)
training_set = np.array(training_set, dtype ='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t', header = None)
test_set = np.array(test_set, dtype ='int')

#GETTING THE NUMBER OF USERS AND MOVIES
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

#CONVERTING THE DATA INTO AN ARRAY WITH USERS IN LINE AND MOVIES IN COLUMNS
def converter(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = converter(training_set)
test_set = converter(test_set)

#CONVERTING THE DATA INTO TORCH TENSORS
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#CREATING THE ARCHITECTURE OF THE NEURAL NETWORK
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#TRAINING THE SAE
nb_epochs = 200
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)  #to add new dimension
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False #so that the gradient is not performed on the target
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

#TESTING THE SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss/s))
