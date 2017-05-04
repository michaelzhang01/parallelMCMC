# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:19:54 2016

Parallel MCMC Sampler Test--Logistic Regression

@author: Michael Zhang
"""

import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from mpi4py import MPI
from scipy.io import loadmat
from scipy.io import savemat
import weiszfeld
from scipy.interpolate import Rbf
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import pdb
import os

def log_one_exp(X):
    X = np.array(X)
    Z = np.empty(X.shape)
    idx = np.where(X < 0.)
    not_idx = np.where(X >= 0.)
    Z[idx] = np.log(1. + np.exp(X[idx]))
    Z[not_idx] = X[not_idx] + np.log(1. + np.exp(-1.*X[not_idx]))
    return(Z)    

class parallelMCMC(object):
    
    def __init__(self,X,Y,beta=None,iters=200, iters2=500, prior_var = 5, 
                 prior_mean = 0., proposal_tune = .25):
        """
        Likelihood of data is:
            Y_i ~ Binomial(Z_i, (1+exp(a+b*X_i))^-1)
        Prior of (a,b) is:
            (a,b) ~ N(prior_mean, prior_var * I)
            
        X: N dim. array of X_i
        Y: N dim. array of Y_i
        Z: N dim. array of Z_i
        a,b : Floats, true values of parameters a and b
        iters: Integer number of subset MCMC iterations
        iters2: Integer number of final MCMC iterations
        prior_mean: 2D array for prior mean of parameters, defined as
            [prior_a, prior_b]
        prior_var: Prior variance of parameters
        proposal_tune: Standard Deviation of Gaussian random walk proposal in 
            subset MCMC
        proposal_tune: Standard Deviation of Gaussian random walk proposal in 
            final MCMC, ignored if random_walk is false.
        random_walk: Boolean, if true then use random walk MH in final MCMC or
            use DPMM proposals in final MCMC otherwise
        """
        self.prior_var = prior_var
        self.prior_mean = prior_mean
        self.proposal_tune = proposal_tune
        self.iters = iters
        self.iters2 = iters2
        self.burnin = self.iters // 2                
#        self.D = 28
#        self.N = int(7e6)
        
        self.comm = MPI.COMM_WORLD
#        self.today = datetime.datetime.today()        
        self.P = self.comm.Get_size()        
        self.rank = self.comm.Get_rank()        
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

#        rank_string = str(int(self.rank))
#        if len(rank_string) < 3:
#            diff = int(3-len(rank_string))
#            fname = 'hepmass_'+ '0'*diff + rank_string
#        dir_fname = os.path.join(os.path.abspath("../data/"),fname)
#        hepmass = np.loadtxt(dir_fname,delimiter=",")
#        self.Y_local = hepmass[:,0].reshape(-1,1).astype(int)
#        self.X_local = hepmass[:,0:]
#        pdb.set_trace()
        self.N = self.comm.bcast(self.N)
        self.D = self.comm.bcast(self.D)
        self.X_local = self.comm.scatter(self.X)
        self.Y_local = self.comm.scatter(self.Y).astype(int)
        self.X = None
        self.Y = None
        self.N_p,D_p = self.X_local.shape
        assert(D_p == self.D)
        assert(self.N_p > 0)
#        self.Y_local = self.Y_local.reshape(self.N_p, -1).astype(int)
#        self.Z_local = self.Z_local.reshape(self.N_p, -1).astype(int)
        
        self.beta_local_trace = np.empty((self.iters,self.D))
#        self.a_local_trace = np.empty(self.iters) # holds all MH acceptances
#        self.b_local_trace = np.empty(self.iters)
        self.LL_local_trace = np.empty(self.iters)
        self.beta_local_all_trace = np.empty((self.iters,self.D))
        self.LL_local_all_trace = np.empty((self.iters,1))
        self.MH_prob_local = [0.,1.]
        
        if self.rank == 0:
#            self.mcmc_trace = mcmc_trace
#            T,D = self.mcmc_trace.shape
#            assert(self.iters2 // 2 == T)
            self.true_beta = beta
            print("Initialization Complete")
        else:
            self.iters2 = None
            self.mcmc_trace = None
            self.true_beta = None


            
    def all_combinations(self,any_list):
        return itertools.chain.from_iterable(
            itertools.combinations(any_list, i + 1)
            for i in xrange(len(any_list)))

    def partition_tuple(self):
        base = [int(self.N / self.P) for k in range(self.P)]        
        remainder = self.N % self.P
        assert(len(xrange(remainder)) <= len(base))
        if remainder:
            for r in xrange(remainder):
                base[r] += 1
        assert(sum(base) == self.N)
        assert(len(base) == self.P)
        return(tuple(base))
        
    def displacement(self, partition):
        if self.P > 1:
            displace = np.append([0], np.cumsum(partition[:-1])).astype(int)
        else:
            displace = [0]
        return(tuple(displace))

    def GR(self):
        """
        Calculates Gelman-Rubin statistics for subset MCMC chains
        """
        local_B_a = np.power(self.a_local_trace[self.burnin:]-self.a_local_trace[self.burnin:].mean(),2).sum()
        local_B_b = np.power(self.b_local_trace[self.burnin:]-self.b_local_trace[self.burnin:].mean(),2).sum()
        local_W_a = self.a_local_trace[self.burnin:].var()
        local_W_b = self.b_local_trace[self.burnin:].var()
        
        self.global_B_a = self.comm.reduce(local_B_a)
        self.global_B_b = self.comm.reduce(local_B_b)
        self.global_W_a = self.comm.reduce(local_W_a)
        self.global_W_b = self.comm.reduce(local_W_b)
        if self.rank==0:
            self.global_B_a *= float(self.burnin)/(float(self.P) - 1.)
            self.global_B_b *= float(self.burnin)/(float(self.P) - 1.)
            self.global_W_a /= float(self.P)
            self.global_W_b /= float(self.P)
            self.V_hat_a = ((float(self.burnin)-1.)/(float(self.burnin)))*self.global_W_a
            self.V_hat_a += ((self.P+1.)/(self.P*self.burnin))*self.global_B_a
            self.V_hat_b = ((float(self.burnin)-1.)/(float(self.burnin)))*self.global_W_b
            self.V_hat_b += ((self.P+1.)/(self.P*self.burnin))*self.global_B_b
            self.GR_a = self.V_hat_a / self.global_W_a
            self.GR_b = self.V_hat_b / self.global_W_b
            print("Gelman-Rubin a: %.2f\tGelman-Rubin b: %.2f" % (self.GR_a, self.GR_b))
        else:
            self.GR_a = None
            self.GR_b = None

    def gather_chains(self):
        """
        Pools MCMC chains from processors
        """
        self.accept_chain = np.array(self.comm.allgather(self.beta_local_trace)).reshape(-1,self.D)
        self.accept_burnin = np.array(self.comm.allgather(self.beta_local_trace[self.burnin:])).reshape(-1,self.D)

        accept_n, accept_d = self.accept_chain.shape
        self.theta = np.array(self.comm.allgather(self.beta_local_all_trace[self.burnin:])).reshape(-1,self.D)
        self.theta_n, self.theta_d = self.theta.shape
        local_LL_eval = np.array([self.local_LL(self.theta[i]) for i in xrange(self.theta_n)])
        local_accept_LL = np.array([self.local_LL(self.accept_chain[i]) for i in xrange(accept_n)])
        self.LL_approx = self.comm.reduce(local_LL_eval)
        self.LL_accept_approx = self.comm.reduce(local_accept_LL)
#        self.theta_mean_burnin = np.array((self.comm.gather(self.a_local_trace[self.burnin:].mean()),
#                                           self.comm.gather(self.b_local_trace[self.burnin:].mean()))).T
        if self.rank==0:
            self.theta_mean_burnin = self.accept_burnin.mean(axis=0)
            self.theta_cov_burnin = np.cov(self.accept_burnin.T)
            self.LL_approx_sd = np.sqrt(self.LL_approx.var())
            self.LL_approx_mean = self.LL_approx.mean()
            self.LL_approx -= self.LL_approx_mean
#            self.LL_approx /= self.LL_approx_sd
            self.scaler = StandardScaler()
            self.theta_scale = self.scaler.fit_transform(self.theta)
            self.h_star = np.exp(minimize(self.local_reg_risk,[0]).x[0])
#            self.h_star = np.exp(minimize_scalar(self.local_reg_risk).x)
#            self.h_star = .01

        else:            
            self.theta = None
            self.theta_mean_burnin = None
            self.LL_approx_sd = None
            self.LL_approx_mean = None
            self.LL_gather = None
            self.LL_approx = None
            self.LL_accept_approx = None
            self.scaler = None
            self.theta_scale = None
            self.h_star = None


    def plot_results(self, grid_size=50, fig_folder = "../output/", 
                     plot_LL_estimate=False):
        """
        Plots results of MCMC sampler.
        
        grid_size: Integer, resolution of the likelihood contour plots.
        fig_folder: String, desired destination to save figures
        plot_LL_estimate: Boolean, whether to plot comparison of true
            likelihood contour and estimated likelihood contour at region where
            final MH acceptances are evaluated.
        """
        fname_today = datetime.datetime.today().strftime("%Y-%m-%d-%f") 
        matfile_fname = os.path.abspath(fig_folder + fname_today + "_" + str(self.P) +".mat")
        save_dict = {'local_mcmc':self.accept_chain,
                     'global_mcmc':self.beta_hat_trace,
                     'proposal_global_mcmc':self.proposal_beta_hat_trace,
                     'global_LL':self.LL_hat_trace,
                     'proposal_global_LL':self.proposal_LL_hat_trace,
                     'local_LL':self.LL_accept_approx}
        savemat(matfile_fname,save_dict)
         
                    
    def sample(self):
        for it in xrange(self.iters):
            self.local_ess(it)
            print("P: %i\tIter: %i\tLL: %.2f\tMH Acceptance: %.2f\tBeta: %s" % (self.rank,it,self.LL_local_trace[it],self.MH_prob_local[0]/self.MH_prob_local[1],self.beta_local_trace[it]))
                
        self.comm.barrier()        
        self.gather_chains()  

        if self.rank==0:            
            self.dp = BayesianGaussianMixture(n_components = 100, max_iter=5000)
            self.dp.fit(self.theta)
            self.beta_hat_trace = np.empty((self.iters2,self.D))
            self.proposal_beta_hat_trace = np.empty((self.iters2,self.D))
            self.LL_hat_trace = np.empty((self.iters2,1))
            self.proposal_LL_hat_trace = np.empty((self.iters2,1))
            self.MH_prob_final = [0.,1.]
            for it in xrange(self.iters2):
                self.final_MH(it)
                print("Iter: %i\tLL: %.2f\tMH Acceptance: %.2f\tBeta: %s" % (it,((self.LL_hat_trace[it])+self.LL_approx_mean),self.MH_prob_final[0]/self.MH_prob_final[1],self.beta_hat_trace[it]))
                print("Proposal LL: %.2f\tProposal Beta: %s" % (((self.proposal_LL_hat_trace[it])+self.LL_approx_mean),self.proposal_beta_hat_trace[it]))
#            self.LL_hat_trace *= self.LL_approx_sd
            self.LL_hat_trace += self.LL_approx_mean
            self.proposal_LL_hat_trace += self.LL_approx_mean
            self.plot_results()
#            final_RMSE = np.sqrt(np.mean((self.beta_hat_trace[self.iters2//2:]- self.mcmc_trace)**2))
#            posterior_concentration = np.sqrt(np.sum((self.beta_hat_trace[self.iters2//2:] - self.true_beta)**2) / np.sum((self.mcmc_trace - self.true_beta)**2))
#            print("Final RMSE: %.2f\t, Concentration Ratio: %.2f" % (final_RMSE,posterior_concentration))
#            fname_today = datetime.datetime.today()        
#            output_str = "../output/" + "RMSE_R"  + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".txt"            
#            np.savetxt(output_str, [final_RMSE, posterior_concentration])

        else:
            self.dp = None
            self.beta_hat_trace = None
            self.a_hat_trace = None
            self.b_hat_trace = None
            self.LL_hat_trace = None
            self.MH_prob_final = None

    def local_ess(self,it):
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
        self.beta_local_all_trace[it] = proposal_beta
#        self.a_local_trace[it] = proposal_a
#        self.b_local_trace[it] = proposal_b
#        self.a_local_all_trace[it] = proposal_a
#        self.b_local_all_trace[it] = proposal_b
        
                
    def local_MH(self,it):
        s_d = (2.4**2)/self.D
        if it == 0:
            self.beta_local_trace[it] = np.random.normal(loc = self.prior_mean, scale = np.sqrt(self.prior_var), size=self.D)
            self.beta_local_all_trace[it] = np.copy(self.beta_local_trace[it])
#            assert(self.local_LL(self.beta_local_trace[it]) == self.local_LL2(self.beta_local_trace[it]))
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
#                proposal_chol = np.linalg.cholesky(proposal_cov)
#                proposal_beta = np.dot(np.random.normal(size=(self.D)),proposal_chol) + self.beta_local_trace[it-1]
                proposal_beta = np.random.multivariate_normal(self.beta_local_trace[it-1],proposal_cov)
                proposal_LL = self.local_LL(proposal_beta)
                while np.isneginf(proposal_LL)  or np.isnan(self.LL_local_trace[it]):
#                    proposal_beta = np.dot(np.random.normal(size=(self.D)),proposal_chol) + self.beta_local_trace[it-1]
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

    def full_LL(self,theta):
        LL = (1. - self.Y)*(np.dot(self.X,theta))
        LL -= np.log(1. + np.exp(np.dot(self.X,theta)))
        LL = LL.sum()
        sigma = self.prior_var*np.eye(self.D)
        theta_diff = (theta-self.prior_mean)
        LL += -.5*np.dot(np.dot(theta_diff.T,sigma),theta_diff)
        return(LL)                
        
#    def local_LL(self,theta):
#        LL = (self.Y_local.flatten())*(np.dot(self.X_local,theta))
#        LL -= (self.Y_local.flatten())*np.log(1. + np.exp(np.dot(self.X_local,theta)))
#        LL += (self.Y_local.flatten()-1.)*np.log(1. + np.exp(np.dot(self.X_local,theta)))
#        LL = LL.sum()
#        sigma = self.prior_var*np.eye(self.D)
#        theta_diff = (theta-self.prior_mean)
#        LL += -.5*np.dot(np.dot(theta_diff.T,sigma),theta_diff)
#        return(LL)
        
    def local_LL(self,theta):        
        LL = np.dot(self.X_local,theta)*self.Y_local.flatten()
        LL -= log_one_exp(np.dot(self.X_local,theta))
#        LL = LL.sum()
#        sigma = (1./self.prior_var)*np.eye(self.D)
#        theta_diff = (theta-self.prior_mean)
#        LL += -.5*np.dot(np.dot(theta_diff.T,sigma),theta_diff)
        return(LL.sum())

        
    def final_MH(self,it):
        if it == 0:
            self.LL_hat_trace[it] = self.predict_LL(self.accept_burnin[-1])
            self.beta_hat_trace[it] = np.copy(self.accept_burnin[-1])
        else:
            proposal_theta = self.dp.sample()[0][0]
            proposal_LL = self.predict_LL(proposal_theta) 
#            accept_prob = ((self.LL_approx_sd*proposal_LL)+self.LL_approx_mean) - ((self.LL_approx_sd*self.LL_hat_trace[it-1])+self.LL_approx_mean) 
            accept_prob = proposal_LL - self.LL_hat_trace[it-1]
            print("Likelihood Ratio: %.2f" % (proposal_LL - self.LL_hat_trace[it-1]))
            accept_prob += self.dp.score(self.beta_hat_trace[it-1]) - self.dp.score(proposal_theta)            
            print("Proposal Ratio: %.2f" % (self.dp.score(self.beta_hat_trace[it-1]) - self.dp.score(proposal_theta)))
            accept_prob += stats.multivariate_normal.logpdf(proposal_theta, self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D)) - stats.multivariate_normal.logpdf(self.beta_hat_trace[it-1], self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D))
            print("Prior Ratio: %.2f" % (stats.multivariate_normal.logpdf(proposal_theta, self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D)) - stats.multivariate_normal.logpdf(self.beta_hat_trace[it-1], self.prior_mean*np.ones(self.D), self.prior_var*np.eye(self.D))))
            u = np.log(np.random.uniform())
            self.MH_prob_final[1] +=1.
            self.proposal_beta_hat_trace[it] = proposal_theta
            self.proposal_LL_hat_trace[it] = proposal_LL
#            pdb.set_trace()
            if u < accept_prob:
                self.beta_hat_trace[it] = proposal_theta
                self.LL_hat_trace[it] = proposal_LL
                self.MH_prob_final[0] +=1.
            else:
                self.beta_hat_trace[it] = np.copy(self.beta_hat_trace[it-1])
                self.LL_hat_trace[it] = self.LL_hat_trace[it-1]
        
#    def predict_LL(self, theta_star, h=.0005):
#        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-theta_star,(self.theta-theta_star)**2))
#        W_i = h*rbf_kernel(theta_star.reshape(1,-1), self.theta, gamma=h) * np.eye(self.theta_n)
#        RW = np.dot(R_i.T, W_i)
#        RWR_inv = np.linalg.inv( np.dot(RW,R_i))
#        RWLL = np.dot(RW, self.LL_approx)
#        beta_hat = np.dot(RWR_inv, RWLL)
#        return(beta_hat[0])
#
#    def local_reg_CV(self,h=.0005):
#        """
#        Objective function to optimize bandwidth parameter for LOOCV of local
#        regression model, very slow!
#        """
#        CV = 0.        
#        for i in xrange(self.theta_n):
#            R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-self.theta[i],(self.theta-self.theta[i])**2))
#            W_i = rbf_kernel(self.theta[i].reshape(1,-1), self.theta, gamma=np.exp(h)) * np.eye(self.theta_n)
#            RW = np.dot(R_i.T, W_i)
#            RWR_inv = np.linalg.inv( np.dot(RW,R_i))
#            RWLL = np.dot(RW, self.LL_approx)
#            beta_hat = np.dot(RWR_inv, RWLL)
#            H_ii = np.dot(np.dot(R_i,RWR_inv),RW)[i,i]
#            Y_hat = beta_hat[0]
#            CV += ((self.LL_approx[i] - Y_hat) / (1.-H_ii))**2
#        return(CV)

    def predict_LL(self, theta_star, h=.0005):
        theta_star_scale = self.scaler.transform(theta_star.reshape(1,-1))
#        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-theta_star,(self.theta-theta_star)**2))
#        W_i = h*rbf_kernel(theta_star.reshape(1,-1), self.theta.reshape(-1,1), gamma=h) * np.eye(self.theta_n)
        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta_scale-theta_star_scale,(self.theta_scale-theta_star_scale)**2))
        W_i = (h/float(self.theta_n))*rbf_kernel(theta_star_scale.reshape(-1,self.D), self.theta_scale.reshape(-1,self.D), gamma=h) * np.eye(self.theta_n)
        RW = np.dot(R_i.T, W_i)
        RWR_inv = np.linalg.inv(np.dot(RW,R_i))
        RWLL = np.dot(RW, self.LL_approx)
        beta_hat = np.dot(RWR_inv, RWLL)
        return(beta_hat[0])


    def local_reg_CV(self,h):
        """
        Objective function to optimize bandwidth parameter for LOOCV of local
        regression model, very slow!
        """
        CV = 0.        
        h=np.exp(h)
        mask = np.ones(self.theta_n).astype(bool)
        for i in xrange(self.theta_n):
            mask[i] = False
            R_i = np.hstack((np.ones((self.theta_n-1,1)), self.theta_scale[mask]-self.theta_scale[i],(self.theta_scale[mask]-self.theta_scale[i])**2))
            W_i = (h/float(self.theta_n-1))*rbf_kernel(self.theta_scale[i].reshape(1,-1), self.theta_scale[mask].reshape(-1,1), gamma=h) * np.eye(self.theta_n-1)
            RW = np.dot(R_i.T, W_i)
            RWR_inv = np.linalg.inv(np.dot(RW,R_i))
            RWLL = np.dot(RW, self.LL_approx[mask])
            beta_hat = np.dot(RWR_inv, RWLL)
            Y_hat = beta_hat[0]
            CV += (self.LL_approx[i] - Y_hat)**2
            mask[i] = True
        CV = np.mean(CV)
        return(CV)

    def local_reg_risk(self,h):
        """
        Objective function to optimize bandwidth parameter for risk estimater of
        local regression
        """
        h = np.exp(h)        
        W = (h/float(self.theta_n))*rbf_kernel(self.theta_scale,gamma=h)
        risk = np.mean((self.LL_approx - np.dot(W,self.LL_approx))**2)/float(self.theta_n)
        sigma_hat = np.mean((self.LL_approx[:-1] - self.LL_approx[1:])**2)
        sigma_hat /= 2.
        risk -= sigma_hat
        risk += 2.*sigma_hat*h/float(self.theta_n)
        return(risk)

if __name__ == '__main__':
    synthetic_data = loadmat("../data/skin.mat")
#    mcmc_trace = np.loadtxt("../data/LR_trace.txt")
#    N=10000
    pmc = parallelMCMC(X=synthetic_data['X'],Y=synthetic_data['Y'])
#    pmc = parallelMCMC(X=synthetic_data['X'][:1000],Y=synthetic_data['Y'][:1000],
#                       beta=synthetic_data['theta'],mcmc_trace=mcmc_trace)
#    pmc = parallelMCMC()
    pmc.sample()