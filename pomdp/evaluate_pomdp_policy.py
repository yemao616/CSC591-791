import csv
from Iohmm_ghmm4 import *
import collections
import numpy as np
import numpy.matlib
import math
import pandas
import sys
import sympy as sp
import mdptoolbox, mdptoolbox.example

def load_data(feature_size):
    # load data set, csv files
    filename = 'reduce.expert.csv'
    original_data = pandas.read_csv(filename)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1

    students_features = ['student', 'priorTutorAction', 'reward']
    select_features = feature_name[start_Fidx: (feature_size + start_Fidx)]
    total_features = students_features + select_features

    data = original_data[total_features]

    # get the mean and variance for each features
    total_features = list(data)
    start_Fidx = 3
    features = total_features[start_Fidx:]
    mu = data[features].mean(axis=0)
    sigma = data[features].var(axis=0)

    distinct_acts = list(data['priorTutorAction'].unique())

    # transfer actions into numbers
    encode_acts = collections.defaultdict(int)
    i = 0
    for act in distinct_acts:
        encode_acts[act] = i
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1

    # transfer data set into observations (mpmath matrix), actions, rewards
    student_list = list(data['student'].unique())
    observations = list()
    actions = list()
    rewards = list()
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()
        if (len(row_list) >= 5):
            observations.append(np.matrix(student_data[features].values))
            actions.append(student_data['priorTutorAction'].tolist())
            rewards.append(student_data['reward'].tolist())

    return [observations, actions, rewards]


def load_model2(fileprefix, feature_size):

    model = Iohmm_ghmm4()
    model.cov_type = 'diagonal'
    model.Dy = feature_size

    filename = fileprefix + 'prior.txt'
    readfile = open(filename, 'rt')
    temp_matrix = list()
    for line in readfile:
        line = line.replace("\n", '')
        line = line.replace('[', '')
        line = line.replace(']', '')
        div = list(filter(None, line.split(' ')))
        div = [float(item) for item in div]
        temp_matrix.append(div)
    readfile.close()

    model.prior = np.matrix(temp_matrix)
    model.Nx = model.prior.shape[1]
    model.Ns = model.prior.shape[0]

    model.RL_transition = np.zeros((model.Ns+1, model.Ns+1, model.Nx))
    model.A = np.zeros((model.Ns, model.Ns, model.Nx))
    model.terminalA = np.zeros((model.Ns, model.Nx))
    model.mu = np.zeros((model.Dy, model.Ns, model.Nx))
    model.sigma = np.zeros((model.Dy, model.Dy, model.Ns, model.Nx))

    for action in range(model.Nx):
        filename = fileprefix + 'transition_'+str(action)+'.txt'
        readfile = open(filename, 'rt')
        temp_matrix = list()

        for line in readfile:
            line = line.replace("\n", "")
            line = line.replace('[', '')
            line = line.replace(']', '')
            div = list(filter(None, line.split(' ')))
            temp_matrix.append(div)

        model.RL_transition[:, :, action]= np.matrix(temp_matrix)
        model.A[:,:,action] = model.RL_transition[0:model.Ns, 0:model.Ns, action]
        model.terminalA[:, action] = model.RL_transition[0:model.Ns, model.Ns, action]
        readfile.close()

        ###############################
        filename = fileprefix + 'mu_' + str(action) + '.txt'
        readfile = open(filename, 'rt')
        mu_state = list()
        for line in readfile:
            line = line.replace("\n", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            div = list(filter(None, line.split(' ')))
            mu_state.append(div)
        model.mu[:,:, action] = np.matrix(mu_state)
        readfile.close()

        ###############################
        filename = fileprefix + 'sigma_' + str(action) + '.txt'
        readfile = open(filename, 'rt')
        s = 0
        for line in readfile:
            line = line.replace("\n", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace(',', '')
            div = list(filter(None, line.split(' ')))
            if s<model.Ns:
                model.sigma[:,:, s, action] = np.diag([float(item) for item in div])
                s += 1
        readfile.close()

    return model


def load_mdp_model(fileprefix):

    filename = fileprefix + 'single_state_QFun.txt'
    readfile = open(filename, 'rt')
    i =0
    single_state_QFun = list()
    actions = list()
    for line in readfile:
        line = line.replace("\n", '')
        line = line.replace('[', '')
        line = line.replace(']', '')
        if i==0:
            actions = list(filter(None, line.split(' ')))
        else:
            div = list(filter(None, line.split(' ')))
            div = [float(item) for item in div]
            single_state_QFun.append(div)
        i += 1
    single_state_QFun = np.matrix(single_state_QFun)
    return [single_state_QFun , actions]


def forward_calculation(model, input, outputSeq):
    T = model.sequence_len(outputSeq)
    Ns = model.Ns                     # remove terminal state
    alpha = np.zeros((T, Ns))
    belief = np.zeros((T, Ns))

    outputlik = model.get_log_emissionP(outputSeq, input)

    # first step
    # prior probability multiply emission probability
    alpha[0, :] = np.squeeze(np.log(model.prior[:, input[0]])) + outputlik[0, :]

    # if sum is zero, which means the value of alpha is too small, we need to do re-scale
    if np.sum(np.exp(alpha[0, :])) == 0.0:
        temp_list = alpha[0, :]
        rescale_value = max(temp_list)
        temp_list = temp_list - rescale_value
        # normalize alpha and obtain hidden state distribution
        belief[0, :] = model.mk_stochastic(np.exp(temp_list))
    else:
        belief[0, :] = model.mk_stochastic(np.exp(alpha[0, :]))

    # non-first step
    for t in range(1, T):
        for s in range(Ns):
            # hidden state distribution multiply transition probability
            value = np.dot(belief[t - 1, :], model.A[:, s, input[t - 1]])
            alpha[t, s] = np.log(value) + outputlik[t, s]

        if np.sum(np.exp(alpha[t, :])) == 0.0:
            temp_list = alpha[t, :]
            rescale_value = max(temp_list)
            temp_list = temp_list - rescale_value
            belief[t, :] = model.mk_stochastic(np.exp(temp_list))
        else:
            belief[t, :] = model.mk_stochastic(np.exp(alpha[t, :]))
    print(belief)
    return belief



def pomdp_ECR(model, Q):
    # Q has the shape Ns * Nx
    Q_function = list()
    for a in range(model.Nx):
        value = Q[:,a].dot(model.prior)
        Q_function.append(value)
    return max(Q_function)

def pomdp_WIS(model, outputs, inputs, Qfun, theta, gamma):
    # remove the terminal state
    state_reward_sum = np.zeros((model.Ns, model.Nx))
    state_reward_pro = np.zeros((model.Ns, model.Nx))
    expectR = np.zeros((model.Ns, model.Nx))

    model.Nseq = len(observations)

    total_seq_count = list()
    for l in range(model.Nseq):
        if model.Nseq == 1:
            xs = inputs
        else:
            xs = inputs[l]  # xs denotes input sequence

        seq_length = np.array([1]*len(xs))

        if len(total_seq_count) != 0:
            if len(total_seq_count) < len(seq_length):
                seq_length[0:len(total_seq_count)] = seq_length[0:len(total_seq_count)] + total_seq_count
                total_seq_count = seq_length
            else:
                total_seq_count[0:len(seq_length)] = total_seq_count[0:len(seq_length)] + seq_length
        else:
            total_seq_count = seq_length


    WIS = 0
    for l in range(model.Nseq):

        print('sequence ' + str(l))
        if model.Nseq == 1:
            ys = outputs
            xs = inputs
            rs = rewards
        else:
            ys = outputs[l].T  # ys denotes output sequence
            xs = inputs[l]  # xs denotes input sequence
            rs = rewards[l]

        belief = forward_calculation(model, xs, ys)

        T = len(xs)

        cumul_logP = 0
        cumul_random = 0
        for t in range(T-1):
            # calculate the probability for each step
            belief_state = np.matrix(belief[t,:], dtype=float)
            values = np.array([0]*model.Nx)

            for x in range(model.Nx):
                values[x] = np.dot(belief_state, Qfun[0:model.Ns, x])   # q function for the state distribution
            values = np.exp(values * theta)
            values = values/ np.sum(values)

            print(values)
            curr_logP = np.log(values[xs[t+1]])

            cumul_logP += curr_logP
            cumul_random += np.log(0.5)

            weight = np.exp(cumul_logP - cumul_random + t*np.log(gamma) - np.log(total_seq_count[t]))
            WIS += weight*rs[t+1]

    print(WIS)
    return(WIS)

def pomdp_policy(belief, Q):
    # belief is hidden state probability distribution
    # Q is the Q function for each hidden state
    # Q has the shape Ns * Nx
    Nx = Q.shape[1]
    Q_function = list()
    for a in range(Nx):
        value = Q[:,a].dot(belief)
        Q_function.append(value)
    value = max(Q_function)
    act = Q_function.index(value)
    return [act, value]

if __name__ == "__main__":

    # load data

    feature_size = 2
    [observations, actions, rewards] = load_data(feature_size)

    # load best model
    fileprefix = 'iohmm_' + str(feature_size) + '/'
    # fileprefix = 'large_iohmm/iohmm_'+str(feature_size)+'/'

    model = load_model2(fileprefix, feature_size)

    # load mdp policy
    [single_QFun, distinct_action] = load_mdp_model(fileprefix)

    # evaluate the Weighted Import Sampling POMDP
    theta = 50
    gamma = 0.9
    pomdp_WIS(model, observations, actions, single_QFun, theta, gamma)