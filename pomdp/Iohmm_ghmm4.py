import sys
import numpy as np
import scipy.stats

# this is the version using mpmath
# this code is the log version. alpha denotes the log of real alpha.
# multivariate gaussian HMM
# scaled version
class Iohmm_ghmm4:

    def __init__(self, outputs=None, Nseq=0, Ns=2, Nx=2, Dy=1, max_iter=1, cov_type='full', scale=True, mu = None, sigma=None):
        self.Dy = Dy                                    # output dimension
        self.Ns = Ns                                    # number of hidden states
        self.Nx = Nx                                    # number of distinct inputs
        self.Nseq = Nseq                                # number of sequences
        self.max_iter = max_iter                        # maximum number of iteration
        self.log_Z = 0                                  # log-likelihood
        self.cov_type = cov_type
        self.scale = scale                              # scale the alpha
        self.weight = 0

        if 40 < Dy <= 50:
            self.converge_weight = self.Nseq
        elif 30 < Dy <= 40:
            self.converge_weight = self.Nseq * 0.3
        elif 20 < Dy <= 30:
            self.converge_weight = self.Nseq * 0.2
        elif 10 < Dy <= 20:
            self.converge_weight = self.Nseq * 0.1
        else:
            self.converge_weight = self.Nseq * 0.05

        # initial guess of parameters
        self.prior = np.random.rand(Ns, Nx) + np.full((Ns, Nx), 0.0001)  # start probability with non-zero elements
        self.prior = self.mk_stochastic(self.prior, row_sum=1)

        # RL_transition contain transition among hidden states as well as transition between hidden state with end state
        self.RL_transition = np.zeros((Ns+1, Ns+1, Nx))
        for x in range(self.Nx):
            self.RL_transition[0:Ns, :, x] = (np.random.rand(Ns, Ns+1) + np.full((Ns, Ns+1), 0.0001))
            self.RL_transition[0:Ns, :, x] = self.mk_stochastic(self.RL_transition[0:Ns, :, x], row_sum=0)
            self.RL_transition[Ns, Ns, x] = 1.0

        self.A = np.zeros((Ns, Ns, Nx))
        self.A = self.RL_transition[0:Ns, 0:Ns, :]
        self.terminalA = np.squeeze(self.RL_transition[0:Ns, Ns, :])

        # multivariate gaussian
        # observations are depended on both hidden state and actions
        if Dy > 1:
            self.mu = np.random.rand(Dy, Ns, Nx)  # state probability, mu
            if mu is not None:
                for s in range(Ns):
                    for x in range(Nx):
                        self.mu[:, s, x] = np.random.rand(1, Dy) + np.array(mu[x])
            else:
                for s in range(Ns):
                    for x in range(Nx):
                        self.mu[:, s, x] = np.random.rand(1, Dy)


            self.sigma = np.zeros((Dy, Dy, Ns, Nx))  # state probability, sigma
            # make sure that sigma is a positive semi-definite matrix (covariance matrix)
            if self.cov_type == 'full':
                for s in range(Ns):
                    for x in range(Nx):
                        temp = np.random.rand(Dy, Dy)*10 + np.full((Dy, Dy), 1.0)
                        self.sigma[:, :, s, x] = np.dot(temp.transpose(), temp)

            elif self.cov_type == 'diagonal':
                # generate a singular covariance matrix
                for s in range(Ns):
                    for x in range(Nx):
                        temp = np.random.rand(Dy, 1)*10 + np.full((Dy, 1), 1.0)
                        self.sigma[:, :, s, x] = np.multiply(np.identity(Dy), temp)
        else:
            self.mu = np.random.rand(Ns, Nx)
            self.sigma = np.random.rand(Ns, Nx)

    def copy(self):
        new_model = Iohmm_ghmm4(Ns=self.Ns, Dy=self.Dy, Nx = self.Nx)

        new_model.log_Z = self.log_Z
        new_model.Nseq = self.Nseq
        new_model.Dy = self.Dy
        new_model.Ns = self.Ns
        new_model.Nx = self.Nx
        new_model.max_iter = self.max_iter
        new_model.cov_type = self.cov_type
        new_model.prior = self.prior
        new_model.terminalA = self.terminalA

        for x in range(self.Nx):
            new_model.A[:,:,x] =  self.A[:,:,x]
            new_model.RL_transition[:,:,x] = self.RL_transition[:,:,x]
            new_model.mu[:, x] = self.mu[:, x]
            for s in range(self.Ns):
                new_model.sigma[:,:, s, x] = self.sigma[:,:,s,x]

        return new_model

    def mk_stochastic(self, T, row_sum = None):
        # the argument is a stochastic matrix, i.e., the sum over the last dimension is 1.
        # % If T is a vector, it will sum to 1.
        # If T is a matrix, each column will sum to 1. sum_i T[i, j] = 1
        # If T is a 3D array, T[i,j,k],  k is idx of a matrix, then sum_j T(i, j, k) = 1 for all j, k.
        T1 = np.squeeze(T)
        if row_sum is None:
            if (len(T1.shape) == 1):
                T1 = T1 / np.sum(T1)
            else:
                T2 = np.sum(T1, axis=0)
                T1 = T1 / np.sum(T2)
        else:
            if (len(T1.shape) == 1):
                T1 = T1/np.sum(T1)

            if row_sum == 1 and (len(T1.shape) == 2):
                # each row will sum to 1 for each column
                T1 = np.divide(T1, np.sum(T1, axis=0))

            elif row_sum == 0 and (len(T1.shape) == 2):
                # each column will sum to 1 for each row
                T1 = np.divide(T1.transpose(), np.sum(T1, axis=1))
                T1 = T1.transpose()

            elif (len(T1.shape) == 1):
                T1 = T1 / np.sum(T1)

            else:
                print("high dimension, can't normalize")
                T1 = T
        return T1

    def sequence_len(self,ys):
        if self.Dy > 1:
            Ny = ys.shape[1]
        else:
            Ny = len(ys)
        return Ny

    def log_normpdf(self, x, mean, var):
        denom = (2 * np.pi * var) ** .5
        num = -(x - mean)** 2 / (2 * var)
        return (num - np.log(denom) + self.weight)

    def log_multigaussian_pro(self, y, mu, sigma):
        # y has size (dy * 1), x is 1*1, mu has size (dy * 1), sigma has size (dy * dy)
        # logP should has size Ns * T
        logP = 0
        if self.cov_type == 'full':
            mu = np.array(np.squeeze(mu))
            sigma = np.array(np.squeeze(sigma))

            var = scipy.stats.multivariate_normal(mean = mu, cov = sigma)
            logP = np.log(var.pdf(np.array(y)))

        elif self.cov_type == 'diagonal':
            logP=0
            for d in range(y.shape[0]):
                # var = scipy.stats.norm(mu[d], np.sqrt(sigma[d,d]))
                #logP += np.log(var.pdf(np.array(y[d])))
                var = self.log_normpdf(y[d], mu[d], sigma[d,d])
                logP += var
        return logP

    def get_log_emissionP(self, ys, xs):
        # ys has size Dy * T,
        # xs has size 1 * T
        # logP should have size  T * Ns

        T = self.sequence_len(ys)
        logP = np.full((T, self.Ns), -np.inf)  # log probability for multivariate gaussian

        if(self.Dy==1):
            for t in range(T):
                logP[t, :] = self.log_normpdf(ys[t], self.mu[:, xs[t]], self.sigma[:, xs[t]])
        else:
            for t in range(T):
                for z in range(self.Ns):
                    logP[t, z] = self.log_multigaussian_pro(ys[:,t], self.mu[:, z, xs[t]], self.sigma[:,:, z, xs[t]])
        return logP


    def forward_backward(self, ys, xs):
        # ys has size (Dy * T) xs has size (1 * T)
        # alpha has size (T * Ns), beta has size (T * Ns), log_emissionP has size (1 * T)
        # both alpha and beta are log version
        # alpha, beta, gamma has size (T * Ns)

        T = self.sequence_len(ys)
        Nx = len(xs)
        if (T != Nx):
            print("input error, not match")

        ###################################################
        ## 1. Calculate emission probability
        log_emissionP = self.get_log_emissionP(ys, xs)

        ###################################################
        # 2. Forward calculation
        alpha = np.full((T, self.Ns), -np.inf)
        alpha_scale = np.array([0]*T)

        # first step: alpha[z0] = p(y0, z0 | x0) = p(z0 | x0) * p(y0| z0, x0)
        # alpha0 = 1, alpha[1] = alpha0 * pro(y1|z1)
        alpha[0, :] = np.log(self.prior[:, xs[0]]) + log_emissionP[0, :]

        if self.scale:

            value = np.sum(np.exp(alpha[0, :]))
            if value == 0.0:
                # which means value of alpha[t,:] are so small we need to rescale
                temp_list = alpha[0, :]
                rescale_value = max(temp_list)
                temp_list = temp_list - rescale_value
                alpha_scale[0] = np.sum(np.exp(temp_list)) + rescale_value
            else:
                alpha_scale[0] =  np.log(value)

            alpha[0, :] = alpha[0,:] - alpha_scale[0]

        # non-first step
        for t in range(1, T):
            for s in range(self.Ns):
                value_list = alpha[t-1, :] + np.log(self.A[:, s, xs[t]])
                alpha[t, s] = np.log(np.sum(np.exp(value_list))) + log_emissionP[t,s]

            if self.scale:
                value = np.sum(np.exp(alpha[t, :]))
                if value == 0.0:
                    # which means value of alpha[t,:] are so small we need to rescale
                    temp_list = alpha[t,:]
                    rescale_value = max(temp_list)
                    temp_list = temp_list - rescale_value
                    alpha_scale[t] = np.sum(np.exp(temp_list)) + rescale_value
                else:
                    alpha_scale[t] = np.log(value)

                alpha[t,:] = alpha[t,:] - alpha_scale[t]

        alpha_end = alpha[T-1, :] + np.log(self.terminalA[:, xs[T-1]])
        alpha_end = np.log(np.sum(np.exp(alpha_end)))

        if self.scale:
            log_Z = np.sum(alpha_scale) + alpha_end
        else:
            log_Z = alpha_end

        ###################################################
        # 3. Backward calculation
        beta = np.full((T, self.Ns), -np.inf)
        beta_end = 0
        beta[T - 1, :] = np.log(self.terminalA[:, xs[T - 1]]) + beta_end

        if self.scale:
            # last step: beta[T-1,]
            beta[T-1, :] = beta[T-1,:] - alpha_scale[T-1]
        else:
            # last step
            beta[T-1, :] = np.log(self.terminalA[:, xs[T - 1]]) + beta_end


        # non-last step
        for t in range(T-2, -1 , -1):
            for s in range(self.Ns):
                value_list = beta[t+1, :] + np.log(self.A[s,:, xs[t+1]]) + log_emissionP[t+1, :]
                beta[t, s] = np.log(np.sum(np.exp(value_list)))

            if self.scale:
                beta[t, :] = beta[t,:] - alpha_scale[t]

        beta0 = beta[0, :] + log_emissionP[0, :] + np.log(self.prior[:, xs[0]])

        if self.scale:
            other_Z = np.sum(alpha_scale) + np.log(np.sum(np.exp(beta0)))
        else:
            other_Z = np.log(np.sum(np.exp(beta0)))

        if abs(log_Z - other_Z) > 1**-6:
            print('forward backward error')

        ###################################################
        # 4. calculate { gamma, xi_sum } for current sequence

        # calculate gamma(i, t) = P(S(t)=i | Y(1:T),X(1:T))
        # gamma has size (T * Ns)
        if self.scale:
            # gamma = alpha * beta * scale
            gamma = (alpha + beta) + (alpha_scale*np.ones((self.Ns, T))).transpose()
            gamma = np.exp(gamma)
        else:
            gamma = (alpha + beta) - ( log_Z * np.ones(T, self.Ns))
            gamma = np.exp(gamma)

        # calculate xi(i,j,t) = P(S(t-1)=i, S(t)=j | Y(1:T),X(1:T))
        #                     = emissionP(yt) * alpha(t-1,i) * beta(t,j) * Aij(xt) / L
        # xi_sum is the probability based instead of log-P

        # xi_end is the expected transition from hidden state to terminal
        if self.scale:
            xi_end = np.exp(alpha[T-1, :] + beta_end + np.log(self.terminalA[:, xs[T-1]]))
        else:
            xi_end = np.exp(alpha[T-1, :] + beta_end - log_Z)

        xi_sum = np.zeros((self.Ns, self.Ns, self.Nx))
        for t in range(T-1):
            for s in range(self.Ns):
                xi = alpha[t,s] + beta[t+1, :] + log_emissionP[t+1, :] + np.log(self.A[ s, :, xs[t+1]])
                xi_sum[s, :, xs[t+1]] += np.exp(xi)

        return [alpha, beta, gamma, log_Z, xi_sum, xi_end]


    def printout_result(self):
        print('prior:')
        print(self.prior)

        print('RL Transition matrix')
        for x in range(self.Nx):
            print('action' + str(x))
            for s in range(self.Ns+1):
                print(self.RL_transition[s,:,x])

        print('Mu')
        for x in range(self.Nx):
            print('action' + str(x))
            for d in range(self.Dy):
                print(self.mu[d, :, x])

        print('sigma')
        for x in range(self.Nx):
            print('action' + str(x))
            for s in range(self.Ns):
                print('state' + str(s))
                value_list = list()
                for d in range(self.Dy):
                    value_list.append(float(self.sigma[d, d, s, x ]))
                print(value_list)
        print("likelihood: " + str(self.log_Z))


    def ess_ghmm(self, inputs, outputs):
        # Expected Sufficient Statistics for mhmm
        # exp_trans[i,j,x] = P(z_t | z_t-1, x_t)
        # exp_visits0[k, x]: P(z_1 |start, Obs, x_1)
        # exp_visitsT[k, x]: P(terminal |z_(T-1)=k, Obs, x_{T-1}) = alpha(T,i) * 1 * terminalA(i, x{T-1}) /L
        # gamma is the log version
        # gamma_sum[z, k] = sum_l sum_t_{xt=k} P(z = k | Y(1:T), X(1,T)) x is the input signal
        # gamma_y[k, :, z] =  sum_l sum_t_{xt=k} gamma(z,k) * Y(:,t,l) / gamma_sum[:,k]
        # gamma_yy [k, z, :, :] = sum_l sum_t_{xt=k} gamma(z,k) * (Y(:,t,l)' * Y(:,t,l)

        # initialization
        exp_trans = np.zeros((self.Ns, self.Ns, self.Nx))
        exp_visits0 = np.zeros((self.Ns, self.Nx))
        exp_visitsT = np.zeros((self.Ns, self.Nx))
        gamma_sum = np.zeros((self.Ns, self.Nx))
        gamma_y = np.zeros((self.Dy, self.Ns, self.Nx))
        gamma_yy = np.zeros((self.Dy, self.Dy, self.Ns, self.Nx))
        gamma_yTy = np.zeros((self.Ns, self.Nx))

        total_log_Z = 0
        for i in range(self.Nseq):
        #for i in range(297, self.Nseq):
        #for i in range(10):
            if self.Nseq == 1:
                xs = np.array(inputs)
                ys =np.array(outputs)
            else:
                xs = np.array(inputs[i])  # xs denotes input sequence
                ys = np.array(outputs[i]).transpose()  # ys denotes output sequence

            if self.Dy > 1:
                T = ys.shape[1]
            else:
                T = len(ys)

            [alpha, beta, gamma, current_log_Z, xi_sum, xi_end] = self.forward_backward(ys, xs)
            # gamma, xi_sum, xi_end are probability-based


            # log-likelihood
            total_log_Z += current_log_Z
            #print(i, float(current_log_Z))

            # expected value of transition among hidden states
            for x in range(self.Nx):
                exp_trans[:, :, x] += xi_sum[:, :, x]

            # expected value of transition from last hidden state to terminal state
            exp_visitsT[:, xs[T - 1]] += xi_end

            # expected value of transition from start to first hidden state
            exp_visits0[:, xs[0]] += gamma[0,:]

            for t in range(T):
                for s in range(self.Ns):
                    gamma_sum[s, xs[t]] += gamma[t,s]
                    # gamma_y
                    gamma_y[:, s, xs[t]] += ys[:, t]*gamma[t, s]
                    # gamma_yy
                    gamma_yy[:, :, s, xs[t]] += np.outer(ys[:, t], ys[:, t]) * gamma[t, s]
                    # gamma_yTy
                    gamma_yTy[s, xs[t]] += np.dot(ys[:, t], ys[:, t]) * gamma[t, s]

        return [total_log_Z, exp_trans, exp_visits0, exp_visitsT, gamma_y, gamma_yy, gamma_yTy, gamma_sum]


    def gauss_Mstep(self, w, Y, YY, YTY):
        # YTY is the inner product
        # YY is the outer product
        # w is the weight generated by gamma_sum
        for s in range(self.Ns):
            for x in range(self.Nx):

                self.mu[:, s, x] = Y[:, s, x] / w[s, x]

                if self.cov_type == 'diagonal':
                    SS = (YY[:, :, s, x] / w[s, x]) - np.outer(self.mu[:, s, x], self.mu[:, s, x])
                    self.sigma[:, :, s, x] = np.diag(np.diag(SS)) + np.identity(self.Dy)*0.001

                elif self.cov_type == 'full':
                    self.sigma[:, :, s, x] = (YY[:, :, s, x] / w[s, x]) - np.outer(self.mu[:, s, x], self.mu[:, s, x])
                    self.sigma[:, :, s, x] += np.identity(self.Dy)*0.001

                elif self.cov_type == 'single':
                    value = ((YTY[s, x] / w[s, x]) - np.dot(self.mu[:, s, x], self.mu[:, s, x])) / self.Dy
                    self.sigma[:, :, s, x] = np.identity(self.Dy) * value + np.identity(self.Dy) * 0.001

    def ghmm_em(self, inputs, outputs):
        num_iter = 0
        converge = False
        previous_logZ = -np.inf

        while(num_iter <= self.max_iter and (not converge)):
            # E step
            [log_Z, exp_trans, exp_visits0, exp_visitsT, gamma_y, gamma_yy, gamma_yTy, gamma_sum] = \
                self.ess_ghmm(inputs, outputs)

            self.log_Z = log_Z

            print('log-likelihood: '+str(self.log_Z))

            # M step
            # prior, make sure each probability is nor zero
            self.prior = self.mk_stochastic(exp_visits0, row_sum=1)
            if np.any(self.prior):
                self.prior += np.full((self.Ns, self.Nx), 0.0001)
                self.prior = self.mk_stochastic(self.prior, row_sum=1)

            #terminal probability, transition probability, normalize at the same time
            for x in range(self.Nx):
                for s in range(self.Ns):
                    self.RL_transition[s, 0:self.Ns, x] = exp_trans[s, :, x]
                    self.RL_transition[s, self.Ns, x] = exp_visitsT[s, x]

                    self.RL_transition[s, :, x] = self.mk_stochastic(self.RL_transition[s, :, x])

                    self.A[s, :, x] = self.RL_transition[s, 0:self.Ns, x]
                    self.terminalA[s, x] = self.RL_transition[s, self.Ns, x]

                    # double check whether zero exist in transitions A, terminalA
                    # if zero exists, smooth

            # Update Gaussian parameter
            self.gauss_Mstep(gamma_sum, gamma_y, gamma_yy, gamma_yTy)

            #self.printout_result()

            #if mp.fabs(log_Z - previous_logZ) < self.Nseq*0.1:  # for feature size is 20
            # if mp.fabs(log_Z - previous_logZ) < self.Nseq*0.2:  # for feature size is 30
            #if mp.fabs(log_Z - previous_logZ) < self.Nseq: # for feature size is 50
            if np.abs(log_Z - previous_logZ) < self.converge_weight:
                    converge = True

            previous_logZ = log_Z
            num_iter += 1



if __name__ == "__main__":
    observations = np.ones((2, 10, 6))
    observations[0, :, :] = (np.random.rand(10, 6) + np.full((10, 6), 0.001))
    observations[1, :, :] = (np.random.rand(10, 6) + np.full((10, 6), 0.001))

    actions = list()
    actions.append([0, 1, 0, 1, 0, 1, 1, 1, 0, 1])
    actions.append([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])
    actions = np.array(actions)

    model = Iohmm_ghmm4(observations, Nseq=2, Ns=2, Nx=2, Dy=6, max_iter=200, cov_type='diagonal')

    # Training the model
    model.ghmm_em(actions, observations)