#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:32:47 2020

@author: ziyad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# DATA PREPROCESSING  

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Thompson Sampling
N = 10000   # Number of rounds/rows
D = 10      # Number of ads/columns  
ads_selected = []
numbers_of_rewards_0 = [0] * D
numbers_of_rewards_1 = [0] * D
total_rewards = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for d in range(0, D):
        random_beta = random.betavariate(numbers_of_rewards_1[d] + 1, 
                                         numbers_of_rewards_0[d] + 1)
        if(random_beta > max_random):
            max_random = random_beta
            ad = d
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if (reward == 1):
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_rewards += reward

# VISUALISATION
plt.hist(ads_selected)
plt.title('Histogram of ad selsections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad is selected')
plt.show()
