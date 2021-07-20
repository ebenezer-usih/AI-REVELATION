#THOMPSON SAMPLING

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#IMPLEMENTING THOMPSON SAMPLING
import math
ads_selected = []
N = 10000
d = 10
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#IMPLEMENTING UCB
import random
ads_selected = []
N = 10000
d = 10
number_of_rewards_1 = [0] * d  
number_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 0
    total_reward = total_reward + reward
    
    
#VISUALISING THE RESULT
plt.hist(ads_selected)
plt.title('HISTOGRAM OF ADS SELECTIONS')
plt.xlabel('ADS')
plt.ylabel('FREQUENCY OF SELECTIONS')
plt.show()