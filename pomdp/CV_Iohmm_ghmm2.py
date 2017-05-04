#from Iohmm_ghmm_mpmath import *
from Iohmm_ghmm4 import *
import csv
import collections
import numpy as np
import numpy.matlib
import math
import pandas
import sys

def load_data(filename, feature_size):
    original_data = pandas.read_csv(filename)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1

    students_features = ['student', 'priorTutorAction', 'reward']
    select_features = feature_name[start_Fidx : (feature_size+start_Fidx)]
    total_features = students_features + select_features

    data = original_data[total_features]
    return data

def cross_validation_iohmm_ghmm(feature_size):
    # load data set, csv files
    filename = 'reduce.famd.features.data.csv'

    data = load_data(filename, feature_size)

    # get the mean and variance for each features
    total_features = list(data)
    start_Fidx = 3
    features = total_features[start_Fidx:]
    mu = data[features].mean(axis=0)
    sigma = data[features].var(axis=0)
    mu = mu.tolist()
    sigma = sigma.tolist()

    distinct_acts = list(data['priorTutorAction'].unique())

    # transfer actions into numbers
    encode_acts = collections.defaultdict(int)
    i = 0
    for act in distinct_acts:
        encode_acts[act] = i
        data.loc[data['priorTutorAction']==act, 'priorTutorAction'] = i
        i += 1

    # transfer data set into observations (mpmath matrix), actions, rewards
    student_list = list(data['student'].unique())
    observations = list()
    actions = list()
    rewards = list()
    i = 0
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()
        if (len(row_list) >= 5):
            observations.append(np.matrix(student_data[features].values))
            actions.append(student_data['priorTutorAction'].tolist())
            rewards.append(student_data['reward'].tolist())
        '''
        if i==296:
            print(student)
            print(student_data)
        '''
        i += 1
    # Training model process
    Ns = 2
    Nx = len(distinct_acts)
    Dy = len(features)
    Nseq = len(observations)

    # Cross Validation
    likelihood_list = list()
    model_list = list()
    for k in range(10):
        print("cross-validation: " +str(k))
        # Initialize model
        model = Iohmm_ghmm4(observations, Nseq, Ns, Nx, Dy, max_iter=100,  cov_type='diagonal')

        # Training the model
        model.ghmm_em(actions, observations)

        # model.printout_result()

        model_list.append(model)
        likelihood_list.append(model.log_Z)

    # choose the best model
    best_index = likelihood_list.index(max(likelihood_list))
    print("best likelihood: " + str(max(likelihood_list)))
    model = model_list[best_index]

    model.printout_result()

if __name__ == "__main__":

    featureSize_list = range(5, 1, -1)
    for feature_size in featureSize_list:
        print('feature size: '+str(feature_size))
        cross_validation_iohmm_ghmm(feature_size)