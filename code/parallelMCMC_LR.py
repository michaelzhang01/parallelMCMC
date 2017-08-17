# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:19:54 2016

Parallel MCMC Sampler Test--Logistic Regression

@author: Michael Zhang
"""

import datetime
#import itertools
#import matplotlib.pyplot as plt
import numpy as np
import os
from mpi4py import MPI
from scipy.io import loadmat
from scipy.io import savemat
#import weiszfeld
#from scipy.interpolate import Rbf
#from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture
from scipy import stats
from sklearn.preprocessing import StandardScaler
#from scipy.optimize import minimize
#from scipy.optimize import minimize_scalar
#import pdb
#import os
#from statsmodels.nonparametric.kernel_regression import KernelReg
from llKernelReg import llKernelReg

def log_one_exp(X):
    """
    Numerically stable code to calculate log( 1 + exp(X))

    Parameters:
        X: Numpy array.
    
    """
    X = np.array(X)
    Z = np.empty(X.shape)
    idx = np.where(X < 0.)
    not_idx = np.where(X >= 0.)
    Z[idx] = np.log(1. + np.exp(X[idx]))
    Z[not_idx] = X[not_idx] + np.log(1. + np.exp(-1.*X[not_idx]))
    return(Z)

class parallelMCMC(object):

    def __init__(self,X,Y,theta=None,beta=None,iters=250, iters2=10000, 
                 prior_var = 5, prior_mean = 0.):
        """
        Likelihood of data is:
            Y_i ~ Bernoulli( 1/(1+exp(Beta*X_i)) )
        Prior of Beta is:
            Beta ~ N(prior_mean, prior_var * I)

        X: NxD dim. matrix of X_i
        Y: N dim. array of Y_i
        Beta : D dim. array true values of regression parameters
        iters: Integer number of subset MCMC iterations
        iters2: Integer number of final MCMC iterations
        prior_mean: 2D array for prior mean of parameters, defined as
            [prior_a, prior_b]
        prior_var: Prior variance of parameters
        """
        self.prior_var = prior_var
        self.prior_mean = prior_mean
        self.iters = iters
        self.iters2 = iters2
        self.burnin = self.iters // 2

        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.theta=theta
        if self.comm.rank == 0:
            self.N,self.D = X.shape
            resort_idx = np.random.choice(self.N, replace=False, size=self.N)
            self.X = np.array_split(X[resort_idx,:], self.P)
            self.Y = np.array_split(Y.flatten()[resort_idx], self.P)

        else:
            self.X = None
            self.Y = None
            self.N = None
            self.D = None

            self.LL_eval = None
            self.interpolate_LL = None
            self.LL_grid = None

        self.N = self.comm.bcast(self.N)
        self.D = self.comm.bcast(self.D)
        self.X_local = self.comm.scatter(self.X)
        self.Y_local = self.comm.scatter(self.Y).astype(int)
        self.X = None
        self.Y = None
        self.N_p,D_p = self.X_local.shape
        assert(D_p == self.D)
        assert(self.N_p > 0)

        self.beta_local_trace = np.empty((self.iters,self.D))
        self.LL_local_trace = np.empty(self.iters)
        self.beta_local_all_trace = np.empty((self.iters,self.D))
        self.LL_local_all_trace = np.empty((self.iters,1))
        self.MH_prob_local = [0.,1.]

        if self.rank == 0:
            self.true_beta = beta
            print("Initialization Complete")
        else:
            self.iters2 = None
            self.mcmc_trace = None
            self.true_beta = None

    def gather_chains(self):
        """
        Pools MCMC chains from processors
        """
        self.all_theta = np.array(self.comm.allgather(self.beta_local_trace[self.burnin:])).reshape(-1,self.D)
        self.theta = self.all_theta.reshape(self.P,-1,self.D).mean(axis=0)

        self.theta_n, self.theta_d = self.all_theta.shape
        local_LL_eval = np.array([self.local_LL(self.all_theta[i]) for i in xrange(self.theta_n)])

        self.LL_approx = self.comm.allreduce(local_LL_eval)
        if self.rank==0:
            self.theta_mean_burnin = self.theta.mean(axis=0)
            self.theta_cov_burnin = np.cov(self.theta.T)
            self.LL_approx_sd = np.sqrt(self.LL_approx.var())
            self.LL_approx_mean = self.LL_approx.mean()
            self.LL_approx -= self.LL_approx_mean
            self.scaler = StandardScaler()
            self.theta_scale = self.scaler.fit_transform(self.theta)

        else:
            self.theta = None
            self.theta_mean_burnin = None
            self.LL_approx_sd = None
            self.LL_approx_mean = None
            self.LL_gather = None
            self.LL_approx = None
            self.scaler = None
            self.theta_scale = None


    def plot_results(self, fig_folder = "../output/"):
        """
        Saves local and global MCMC chains

        fig_folder: String, desired destination to save figures
        """
        fname_today = datetime.datetime.today().strftime("%Y-%m-%d-%f")
        matfile_fname = os.path.abspath(fig_folder + fname_today + "_" + str(self.P) +".mat")
        save_dict = {'local_mcmc':self.all_theta,
                     'global_mcmc':self.beta_hat_trace,
                     'global_LL':self.LL_hat_trace,
                     'local_LL':self.LL_approx}
        savemat(matfile_fname,save_dict)


    def sample(self):
        """
        Performs local MCMC sampling and global sampling.
        """
        for it in xrange(self.iters):
            self.local_ess(it)
            print("P: %i\tIter: %i\tLL: %.2f\tMH Acceptance: %.2f\tBeta: %s" % (self.rank,it,self.LL_local_trace[it],self.MH_prob_local[0]/self.MH_prob_local[1],self.beta_local_trace[it]))

        self.comm.barrier()
        self.gather_chains()

        if self.rank==0:
            self.dp = BayesianGaussianMixture(n_components = 100, max_iter=5000)
            self.dp.fit(self.theta)
#            self.LL_reg = KernelReg(exog=self.all_theta,endog=self.LL_approx.reshape(-1,1),var_type='c'*self.D)
            self.LL_reg = llKernelReg(X=self.all_theta,Y =  self.LL_approx.flatten())
            self.beta_hat_trace = np.empty((self.iters2,self.D))
            self.proposal_beta_hat_trace = np.empty((self.iters2,self.D))
            self.LL_hat_trace = np.empty((self.iters2,1))
            self.proposal_LL_hat_trace = np.empty((self.iters2,1))
            self.MH_prob_final = [0.,1.]
            for it in xrange(self.iters2):
                self.final_MH(it)
                print("Iter: %i\tLL: %.2f\tMH Acceptance: %.2f\tBeta: %s" % (it,((self.LL_hat_trace[it])+self.LL_approx_mean),self.MH_prob_final[0]/self.MH_prob_final[1],self.beta_hat_trace[it]))
            self.LL_hat_trace += self.LL_approx_mean
            self.proposal_LL_hat_trace += self.LL_approx_mean
            self.plot_results()

        else:
            self.dp = None
            self.beta_hat_trace = None
            self.a_hat_trace = None
            self.b_hat_trace = None
            self.LL_hat_trace = None
            self.MH_prob_final = None

    def local_ess(self,it):
        """
        Local inference method using eliptical slice sampling.
        
        it: Integer, current iteration of MCMC sampler.
        """
        V = np.random.normal(scale = np.sqrt(self.prior_var), size=self.D)
        u = np.log(np.random.uniform())
        if it == 0:
            self.beta_local_trace[it] = np.random.normal(scale = np.sqrt(self.prior_var), size=self.D)
            log_LL = self.local_LL(self.beta_local_trace[it]) + u
        else:
            log_LL = self.local_LL(self.beta_local_trace[it-1]) + u

        theta = np.random.uniform(0., 2.*np.pi)
        bracket = [theta-(2.*np.pi),theta]

        if it == 0:
            proposal_beta = V*np.sin(theta) + self.beta_local_trace[it]*np.cos(theta)
        else:
            proposal_beta = V*np.sin(theta) + self.beta_local_trace[it-1]*np.cos(theta)

        self.LL_local_trace[it] = self.local_LL(proposal_beta)
        while self.LL_local_trace[it] < log_LL:
            if theta < 0:
                bracket[0] = theta
            else:
                bracket[1] = theta
            theta = np.random.uniform(bracket[0],bracket[1])
            if it == 0:
                proposal_beta = V*np.sin(theta) + self.beta_local_trace[it]*np.cos(theta)
            else:
                proposal_beta = V*np.sin(theta) + self.beta_local_trace[it-1]*np.cos(theta)

            self.LL_local_trace[it] = self.local_LL(proposal_beta)

        self.beta_local_trace[it] = proposal_beta

    def local_MH(self,it):
        """
        Not used
        """
        s_d = (2.4**2)/self.D
        if it == 0:
            self.beta_local_trace[it] = np.random.normal(loc = self.prior_mean, scale = np.sqrt(self.prior_var), size=self.D)
            self.beta_local_all_trace[it] = np.copy(self.beta_local_trace[it])
            self.LL_local_trace[it] = self.local_LL(self.beta_local_trace[it])
            while np.isneginf(self.LL_local_trace[it]) or np.isnan(self.LL_local_trace[it]):
                self.beta_local_trace[it] = np.random.normal(loc = self.prior_mean, scale = np.sqrt(self.prior_var), size=self.D)
                self.beta_local_all_trace[it] = np.copy(self.beta_local_trace[it])
                self.LL_local_trace[it] = self.local_LL(self.beta_local_trace[it])

        else:
            if it <= self.iters // 4:
                proposal_beta = np.random.normal(loc = self.beta_local_trace[it-1],
                                                 scale = self.proposal_tune)
                proposal_LL = self.local_LL(proposal_beta)
                while np.isneginf(proposal_LL)  or np.isnan(self.LL_local_trace[it]):
                    proposal_beta = np.random.normal(loc = self.beta_local_trace[it-1],
                                                 scale = self.proposal_tune)
                    proposal_LL = self.local_LL(proposal_beta)

            else:
                proposal_cov = s_d*(np.cov(self.beta_local_trace[:it].T) + (1e-6)*np.eye(self.D))
                proposal_beta = np.random.multivariate_normal(self.beta_local_trace[it-1],proposal_cov)
                proposal_LL = self.local_LL(proposal_beta)
                while np.isneginf(proposal_LL)  or np.isnan(self.LL_local_trace[it]):
                    proposal_beta = np.random.multivariate_normal(self.beta_local_trace[it-1],proposal_cov)
                    proposal_LL = self.local_LL(proposal_beta)


            accept_prob = proposal_LL - self.LL_local_trace[it-1]
            accept_prob += stats.multivariate_normal.logpdf(proposal_beta, self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D)) - stats.multivariate_normal.logpdf(self.beta_local_trace[it-1], self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D))
            u = np.log(np.random.uniform())
            self.MH_prob_local[1] += 1.
            self.beta_local_all_trace[it] = proposal_beta
            if u < accept_prob:
                self.beta_local_trace[it] = proposal_beta
                self.LL_local_trace[it] = proposal_LL
                self.MH_prob_local[0] +=1.
            else:
                self.beta_local_trace[it] = np.copy(self.beta_local_trace[it-1])
                self.LL_local_trace[it] = np.copy(self.LL_local_trace[it-1])


    def local_LL(self,theta):
        """
        Calculates local likelihood

        theta: D dimensional array, representing parameters of model.
        """
        LL = np.dot(self.X_local,theta)*(1.-self.Y_local.flatten())
        LL -= log_one_exp(np.dot(self.X_local,theta))
        assert(~np.isnan(LL.sum()))
        return(LL.sum())

    def final_MH(self,it):
        min_LL = -1e300
        if it == 0:
            self.beta_hat_trace[it] = self.theta_mean_burnin
            self.LL_hat_trace[it] = self.LL_reg.fit(self.beta_hat_trace[it].reshape(1,-1),min_LL)

        else:
            proposal_theta = self.dp.sample()[0]
            proposal_LL = self.LL_reg.fit(proposal_theta.reshape(1,-1),min_LL)
            accept_prob = proposal_LL.flatten() - self.LL_hat_trace[it-1]
            accept_prob += self.dp.score([self.beta_hat_trace[it-1]]) - self.dp.score(proposal_theta)
            accept_prob += stats.multivariate_normal.logpdf(proposal_theta, self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D)) - stats.multivariate_normal.logpdf(self.beta_hat_trace[it-1], self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D))

            u = np.log(np.random.uniform())
            self.MH_prob_final[1] +=1.
            self.proposal_beta_hat_trace[it] = proposal_theta
            self.proposal_LL_hat_trace[it] = proposal_LL
            if u < accept_prob:
                self.beta_hat_trace[it] = proposal_theta
                self.LL_hat_trace[it] = proposal_LL
                self.MH_prob_final[0] +=1.
            else:
                self.beta_hat_trace[it] = np.copy(self.beta_hat_trace[it-1])
                self.LL_hat_trace[it] = self.LL_hat_trace[it-1]


    def predict_LL(self, theta_star, farValue=-1e300):
        mean = self.LL_reg.fit(theta_star,farValue)
        return(mean)



if __name__ == '__main__':
    synthetic_data = loadmat("../data/skin.mat")
    N,D = synthetic_data['X'].shape
    pmc = parallelMCMC(X=synthetic_data['X'],Y=synthetic_data['Y'], theta=None)
    pmc.sample()