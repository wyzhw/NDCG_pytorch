# -*- coding: utf-8 -*-

# Import Library

import csv
import random
import pandas as pd
import seaborn as sns
data = pd.read_csv("data/ml-100k/u.data",sep="\t",header=None)

# Our dataprocessing as follows
# Only select rating > 3 as a possivtive one

sns.set_theme(style="darkgrid")
ax = sns.countplot(x = data[2])
ax.set(xlabel='Rating score')

"""# Select rating > threshold in movielens dataset"""

def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def insert_user(movie_id, count):
    user_item = []
    movie_id.insert(0, count)
    user_item.append(movie_id)
    return user_item

def create_data(threshold, split_percentage):
    '''Create a data set from the u.data file.
    Only select rating > threshold as a possivtive one
    Split the data into two sets: training and test'''
    total_train = []
    total_test = []
    count = 0
    for user in range(1,max(data[0])+1): # max(data[0])+1
        movie_id = data[(data[0]== user) & (data[2]> threshold)][1].tolist()
        train, test = split_data(movie_id, split_percentage)
        
        if len(test)>1:
            insert_user(train,count)
            insert_user(test,count)
            total_train.append(train)
            total_test.append(test)
            count +=1
   
    with open("data/rating/train.txt", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerows(total_train) 
    with open("data/rating/test.txt", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerows(total_test)

    print("The selected dataset is created! ")