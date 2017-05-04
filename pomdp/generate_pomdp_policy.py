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

def load_model1(fileprefix):
    # load prior infor
    filename = fileprefix + "_prior.txt"
    readfile = open(filename, 'r')
    prior = list()
    for line in readfile:
        line = line.replace("\n", "")
        prior = float(line.split())
        break
    readfile.close()
    prior = np.array(prior)
    Ns = prior.shape[0]

    # load reward
    filename = fileprefix + "_R.txt"
    readfile = open(filename, 'r')
    R = list()
    for line in readfile:
        line = line.replace("\n","")
        div = line.split(" ")
        R.append(float(div))
    R = np.array(R)
    readfile.close()

    if R.shape[0] == Ns:
        Nx = R.shape[1]
    else:
        R = R.transpose()
        Nx = R.shape[1]

    # load transition probability
    A = np.zeros((Nx, Ns, Ns))
    for a in range(Nx):
        filename = fileprefix + "_P" + str(a) + ".txt"
        readfile = open(filename, 'r')
        s = 0
        for line in readfile:
            line = line.replace("\n", "")
            div = line.split(" ")
            A[a, s, :] = np.array(float(div))
            s += 1
        readfile.close()
    return [A, R, prior]

def load_model2(feature_size):
    fileprefix = 'iohmm_'+str(feature_size)+'/'
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
            line = line.replace("\n","")
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
            if s<model.Ns:                         #########modification here
                model.sigma[:, :, s, action] = np.diag([float(item) for item in div])
                s += 1
        readfile.close()

    return model

def load_mdp_model(fileprefix):
    # Q value for each pair of state and action
    filename = fileprefix + 'single_state_MDP.txt'
    readfile = open(filename, 'rt')
    i =0
    single_state_QFun = list()
    actions = list()
    for line in readfile:
        line = line.replace('\n', '')
        if i==0:
            actions = line.split(' ')
        else:
            div = line.split(' ')
            single_state_QFun.append(div)
        i += 1

    return [single_state_QFun, actions]


def forward_calculation(model, input, outputSeq):
    T = model.sequence_len(outputSeq)
    Ns = model.Ns                     # remove terminal state
    alpha = np.zeros((T, Ns))
    belief = np.zeros((T, Ns))

    outputlik = model.get_log_emissionP(outputSeq, input)

    # first step
    alpha[0, :] = np.squeeze(np.log(model.prior[:, input[0]])) + outputlik[0, :]

    if np.sum(np.exp(alpha[0, :])) == 0.0:
        temp_list = alpha[0, :]
        rescale_value = max(temp_list)
        temp_list = temp_list - rescale_value
        belief[0, :] = model.mk_stochastic(np.exp(temp_list))
    else:
        belief[0 ,:] = model.mk_stochastic(np.exp(alpha[0, :]))

    # non-first step
    for t in range(1, T):
        for s in range(Ns):
            value = np.dot(belief[t-1, :], model.A[:, s, input[t-1]])
            alpha[t, s] = np.log(value) + outputlik[t, s]

        if np.sum(np.exp(alpha[t, :])) == 0.0:
            temp_list = alpha[t, :]
            rescale_value = max(temp_list)
            temp_list = temp_list - rescale_value
            belief[t, :] = model.mk_stochastic(np.exp(temp_list))
        else:
            belief[t, :] = model.mk_stochastic(np.exp(alpha[t, :]))

    return belief

def Calculate_expectR(model, outputs, inputs, rewards):
    # remove the terminal state
    state_reward_sum = np.zeros((model.Ns, model.Nx))
    state_reward_pro = np.zeros((model.Ns, model.Nx))

    model.Nseq = len(outputs)
    for l in range(model.Nseq):
        if model.Nseq == 1:
            ys = np.array(outputs).transpose()
            xs = np.array(inputs)
            rs = np.array(rewards)
        else:
            ys = np.array(outputs[l]).transpose()  # ys denotes output sequence
            xs = np.array(inputs[l])               # xs denotes input sequence
            rs = np.array(rewards[l])

        T = len(xs)
        belief = forward_calculation(model, xs, ys)

        for t in range(T-1):
            state_reward_sum[:, xs[t+1]] += rs[t+1] # * belief[t, :]
            state_reward_pro[:, xs[t+1]] += belief[t, :]

    expectR = np.divide(state_reward_sum, state_reward_pro)

    return expectR

# when storing the model, removing terminal state
def store_MDP(fileprefix, model, expectR):
    # write down prior probability
    filename = fileprefix + '_prior.txt'
    writefile = open(filename, 'w')
    value = ""
    for i in range(model.Ns-1):
        value += str(model.prior[i]) + " "
    value = value[0:-1]
    value += "\n"
    writefile.write(value)
    writefile.close()

    # write down transition probability model.A
    for a in range(model.Nx):
        filename = fileprefix + "_P" + str(a) + ".txt"
        writefile = open(filename, 'w')
        for i in range(model.Ns-1):
            value = ""
            for j in range(model.Ns-1):
                value += str(model.A[a, i, j]) + " "
            value = value[0:-1]
            writefile.write(value + "\n")
        writefile.close()

    # write down expected R
    filename = fileprefix + '_R.txt'
    writefile = open(filename, 'w')
    # no terminal state
    for i in range(model.Ns-1):
        value = ""
        for a in range(model.Nx):
            value += str(expectR[i, a]) + " "
        value = value[0:-1]
        writefile.write(value + "\n")
    writefile.close()

    filename = fileprefix +"_obsmat.txt"
    writefile = open(filename, 'w')
    for i in range(model.Dy-1):
        for s in range(model.Ns-1):
            value = ""
            for item in list(model.obsmat[i][s,:]):
                value += str(item)+" "
            value = value[0:-1]
            writefile.write(value+"\n")
        writefile.write("\n")
    writefile.close()

def pomdp_ECR(model, Q):
    # Q has the shape Ns * Nx
    Q_function = list()
    for a in range(model.Nx):
        value = Q[:, a].dot(model.prior)
        Q_function.append(value)
    return max(Q_function)

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

    feature_size = 2
    # load data
    [observations, actions, rewards] = load_data(feature_size)

    # load best model
    model = load_model2(feature_size)

    # calculate the expected reward                                                       [[-15.8352263  -10.78101809]
    expectR = Calculate_expectR(model, observations, actions, rewards)

    print(expectR)
    #fileprefix = "discrete_model"+str(model.Ns)
    #store_MDP(fileprefix, model, expectR)

    RL_expectR = np.zeros((model.Ns+1, model.Nx))

    for x in range(model.Nx):
        RL_expectR[0:model.Ns, x] = expectR[:,x]
    RL_expectR[model.Ns,:]=[0.0]*model.Nx

    #print(RL_expectR)

    RL_transition = np.zeros((model.Nx, model.Ns+1, model.Ns+1))

    for x in range(model.Nx):
        RL_transition[x, model.Ns, model.Ns] = 1
        for s in range(model.Ns):
            RL_transition[x, s, :] = model.mk_stochastic(model.RL_transition[s,:, x])

    print(RL_expectR.shape)
    print(RL_transition.shape)

    pi = mdptoolbox.mdp.QLearning(RL_transition, RL_expectR, 0.9)

    pi.run()
    print(pi.policy)
    print(pi.Q)

    # save pi.Q as 'single_state_QFun.txt'
    
    #ECR = pomdp_ECR(model, pi.Q)
    #print("ECR: "+str(ECR))