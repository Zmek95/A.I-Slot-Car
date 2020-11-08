#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:56:07 2020

@author: ziyad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# DATA PREPROCESSING  

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB
N = 10000   # Number of rounds/rows
D = 10      # Number of ads/columns  
ads_selected = []
no_of_selections = [0] * D
sum_of_rewards = [0] * D
total_rewards = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for d in range(0, D):
        if (no_of_selections[d] > 0):
            average_reward = sum_of_rewards[d]/no_of_selections[d]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/no_of_selections[d])
            upper_bound = average_reward + delta_i
        else: # This branch is entered when no ads are selected
            upper_bound = 1e400 # Infinity
        
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = d
    
    ads_selected.append(ad)
    no_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_rewards += reward

# VISUALISATION
plt.hist(ads_selected)
plt.title('Histogram of ad selsections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad is selected')
plt.show()
