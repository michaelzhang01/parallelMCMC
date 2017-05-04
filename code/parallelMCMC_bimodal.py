# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 09:28:17 2016

Parallel MCMC--Bimodal Example

@author: Michael Zhang
"""

import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys
from mpi4py import MPI
from scipy.io import loadmat
from scipy.interpolate import Rbf
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import foldnorm
from scipy.stats import norm
from scipy.stats import gamma
from scipy.misc import logsumexp
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


class parallelMCMC_bimodal(object):
    
    def __init__(self, X,Z=None, epsilon=None, delta=None, mu=None,
                 sigma=None, pi=None, alpha=1., a=1., b=1., 
                 proposal_sigma = 1., prior_mu=0., prior_sigma=100.,                  
                 prior_obs = 1., iters=50, iters2=10000):
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
#        self.prior_a = a
#        self.prior_b = b
#        self.true_theta = theta
        self.K = 2
        self.dim = 1.
        self.pi = pi
        self.sigma = sigma
        self.mu = mu
        self.epsilon = epsilon
        self.delta = delta
        self.alpha=alpha
        self.prior_a = a
        self.prior_b = b
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_obs = prior_obs
        self.proposal_sigma = proposal_sigma
#        self.scaling = ((2.4)**2)/self.dim
#        self.proposal_tune_2 = proposal_tune_2
#        self.random_walk = bool(random_walk)
#        self.prior_var = prior_var
#        self.prior_mean = np.array(prior_mean)
#        self.proposal_tune = proposal_tune
        self.iters = iters
        self.burnin = self.iters // 2                
        self.iters2 = iters2
        self.comm = MPI.COMM_WORLD
#        self.today = datetime.datetime.today()        
        self.P = self.comm.Get_size()        
        self.rank = self.comm.Get_rank()        
        if self.comm.rank == 0:
            self.X = X.astype(float)
#            self.Y = Y.astype(float)
            self.Z = Z.astype(float)
        else:
            self.X = None
#            self.Y = None
            self.Z = None
#            self.a_grid = None
#            self.b_grid = None
#            self.a_mesh = None
#            self.b_mesh = None
            self.LL_eval = None
            self.interpolate_LL = None
            self.LL_grid = None

        self.X = self.comm.bcast(self.X)
#        self.Y = self.comm.bcast(self.Y)        
        self.true_Z = self.comm.bcast(self.Z) 
        self.N,self.D = self.X.shape
        
        self.part_size_X = tuple([j * self.D for j in self.partition_tuple()])
#        self.part_size_Y = tuple([j for j in self.partition_tuple()])
        self.part_size_Z = tuple([j for j in self.partition_tuple()])
        self.data_displace_X = self.displacement(self.part_size_X)
#        self.data_displace_Y = self.displacement(self.part_size_Y)        
        self.data_displace_Z = self.displacement(self.part_size_Z)        
        self.X_local = np.zeros(self.partition_tuple()[self.rank] * self.D)        
#        self.Y_local = np.zeros((self.partition_tuple()[self.rank]))
        self.true_Z_local = np.zeros((self.partition_tuple()[self.rank]))
#        self.comm.Scatterv([self.Y, self.part_size_Y, self.data_displace_Y, MPI.DOUBLE], self.Y_local)                
        self.comm.Scatterv([self.X, self.part_size_X, self.data_displace_X, MPI.DOUBLE], self.X_local)                    
        self.comm.Scatterv([self.true_Z, self.part_size_Z, self.data_displace_Z, MPI.DOUBLE], self.true_Z_local)

        self.X_local = self.X_local.reshape((-1, self.D))     
        self.N_p,self.D_p = self.X_local.shape
        assert(self.D_p == self.D)
        assert(self.N_p > 0)
        init_dp = KMeans(n_clusters = self.K, max_iter=5000)
        init_dp.fit(self.X_local)
        self.Z_local = init_dp.predict(self.X_local)
        self.Z_local_count = np.bincount(self.Z_local, minlength=self.K)

#        self.Y_local = self.Y_local.reshape(self.N_p, -1).astype(int)
        self.true_Z_local = self.Z_local.reshape(self.N_p, -1).astype(int)
        self.mu_local_trace = np.empty((self.iters,2))
        self.mu_local_all_trace = np.empty((self.iters,self.K))        
        self.sigma_local_trace = np.empty((self.iters,self.K))
        self.sigma_local_all_trace = np.zeros((self.iters,self.K))        
        self.pi_local_trace = np.empty((self.iters, self.K))

        self.theta_local_all_trace = np.empty((self.iters2, self.K*2))
#        self.a_local_trace = np.empty(self.iters) # holds all MH acceptances
#        self.b_local_trace = np.empty(self.iters)
        self.LL_local_trace = np.zeros(self.iters)
#        self.a_local_all_trace = np.empty((self.iters,1)) # holds all MH proposals
#        self.b_local_all_trace = np.empty((self.iters,1))
        self.LL_local_all_trace = np.zeros((self.iters,1))

        self.MH_prob_local = [1.,1.]
        
        if self.rank == 0:
            print("Initialization Complete")
        else:
            self.X = None
#            self.Y = None
#            self.Z = None
            
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
        combined_theta = np.hstack((self.mu_local_trace,self.sigma_local_trace))
#        all_theta = np.hstack((self.mu_local_all_trace,self.sigma_local_all_trace))
        self.accept_chain = np.array(self.comm.allgather(combined_theta)).reshape(-1,self.K*2)
        self.accept_burnin = np.array(self.comm.allgather(combined_theta[self.burnin:])).reshape(-1,self.K*2)

        accept_n, accept_d = self.accept_chain.shape
        self.theta = np.array(self.comm.allgather(combined_theta[self.burnin:])).reshape(-1,self.K*2)
        self.theta_n, self.theta_d = self.theta.shape
        self.pi_burnin = np.array(self.comm.allgather(self.pi_local_trace[self.burnin:])).reshape(-1,self.K).mean(axis=0)
        self.pi_burnin /= self.pi_burnin.sum()
#        local_LL_eval = np.array([norm.logpdf(x,th[z],th[z+1]) for x,z,th in zip(self.X_local,self.Z_local,self.theta)])
#        local_accept_LL = np.array([norm.logpdf(x,th[z],th[z+1]) for x,z,th in zip(self.X_local,self.Z_local,self.accept_chain)])

        local_LL_eval = np.array([np.log(np.sum([np.exp(np.log(self.pi_burnin[k]) + norm.logpdf(self.X_local,theta[k],theta[k+2])) for k in xrange(self.K)],axis=0)).sum() for theta in self.theta])        
        self.LL_approx = self.comm.reduce(local_LL_eval)
#        pdb.set_trace()
#        self.LL_accept_approx = self.comm.reduce(local_accept_LL)
        self.Z_count_total = self.comm.reduce(self.Z_local_count)

        if self.rank==0:
            self.theta_mean_burnin = self.accept_burnin.mean(axis=0)
            self.theta_cov_burnin = np.var(self.accept_burnin,axis=0)
            self.LL_approx_sd = np.sqrt(self.LL_approx.var())
            self.LL_approx_mean = self.LL_approx.mean()
            self.LL_approx -= self.LL_approx_mean
#            self.LL_approx /= self.LL_approx_sd
            self.scaler = StandardScaler()
            self.theta_scale = self.scaler.fit_transform(self.theta)
#            self.h_star = np.exp(minimize_scalar(self.local_reg_risk).x)            
            self.h_star = np.exp(minimize(self.local_reg_risk,[0]).x[0])            
#            self.h_star = .05
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
        if self.rank == 0:
            fname_today = datetime.datetime.today()        

            self.local_plot_f, self.local_plot_arr = plt.subplots(2,3, figsize=(15.75,8))            
            X_sort = np.argsort(self.X.flatten())
            X_local_sort = np.argsort(self.X_local.flatten())
            true_lik = np.sum([np.exp(np.log(self.pi[k]) + norm.logpdf(self.X.flatten(), self.mu[k], self.sigma[k])) for k in xrange(self.K)],axis=0)
#            post_lik = np.sum([np.exp(norm.logpdf(self.X.flatten(), self.theta_hat_trace[:,k].mean(), self.theta_hat_trace[:,k+self.K].mean())) for k in xrange(self.K)],axis=0)
            self.local_plot_arr[0,0].plot(self.X.flatten()[X_sort], true_lik[X_sort],'b-')

            for tr in xrange(0,self.P*self.burnin, self.burnin):
                post_lik = np.sum([np.exp( np.log((self.Z_local_count[k]+self.alpha)/(self.N_p+self.alpha)) + norm.logpdf(self.X_local.flatten(), self.accept_burnin[tr:tr+self.burnin,k].mean(), self.accept_burnin[tr:tr+self.burnin,k+self.K].mean().mean())) for k in xrange(self.K)],axis=0)
                self.local_plot_arr[0,0].plot(self.X_local.flatten()[X_local_sort], post_lik[X_local_sort],'g--')
            
            self.local_plot_arr[0,0].set_title("Density Plot")
            self.local_plot_arr[0,0].set_xlabel("X")
            self.local_plot_arr[0,0].set_ylabel("Density")            
            self.local_plot_arr[0,0].legend(["True","Subset"])

            mu_counter = 0
            sigma_counter = 0
            for param in xrange(self.accept_chain.shape[1]):
                if param < self.K:
                    mu_counter += 1
                    self.local_plot_arr[1,0].plot(self.accept_chain[:,param],'-',label="mu_"+str(mu_counter))                                
                else:
                    sigma_counter += 1 
                    self.local_plot_arr[1,0].plot(self.accept_chain[:,param],'-.',label="sigma_"+str(sigma_counter))

            self.local_plot_arr[1,0].set_title("Traceplot")
            self.local_plot_arr[1,0].set_xlabel("Iteration")
            self.local_plot_arr[1,0].set_ylabel("Parameter")            
            self.local_plot_arr[1,0].legend(loc='lower center',ncol=2,bbox_to_anchor=(0., -.45, 1., .102),mode="expand")

            self.local_plot_arr[0,1].hist(self.accept_burnin[:,0])
            self.local_plot_arr[0,1].set_title("Traceplot mu_1")
            self.local_plot_arr[0,1].set_xlabel("mu_1")

            self.local_plot_arr[0,2].hist(self.accept_burnin[:,1])
            self.local_plot_arr[0,2].set_title("Traceplot mu_2")
            self.local_plot_arr[0,2].set_xlabel("mu_2")

            self.local_plot_arr[1,1].hist(self.accept_burnin[:,2])
            self.local_plot_arr[1,1].set_title("Traceplot sigma_1")
            self.local_plot_arr[1,1].set_xlabel("sigma_1")

            self.local_plot_arr[1,2].hist(self.accept_burnin[:,3])
            self.local_plot_arr[1,2].set_title("Traceplot sigma_2")
            self.local_plot_arr[1,2].set_xlabel("sigma_2")
            
            self.local_plot_f.subplots_adjust(hspace=.3, wspace=.3)

            local_plot_fname = fig_folder + "bimodal_subset_MH_"  + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".png"
            self.local_plot_f.savefig(os.path.abspath(local_plot_fname), dpi=600, format='png', bbox_inches='tight')

            self.global_plot_f, self.global_plot_arr = plt.subplots(2,3, figsize=(15.75,8))            
            X_sort = np.argsort(self.X.flatten())
#            true_lik = np.sum([np.exp(norm.logpdf(self.X.flatten(), self.mu[:,k], self.sigma[:,k])) for k in xrange(self.K)],axis=0)
            final_pi = self.Z_count_total + self.alpha
            final_pi /= final_pi.sum()
            final_pi = np.log(final_pi)

            post_lik = np.sum([np.exp( final_pi[k] + norm.logpdf(self.X.flatten(), self.theta_hat_trace[:,k].mean(), self.theta_hat_trace[:,k+2].mean())) for k in xrange(self.K)],axis=0)
            self.global_plot_arr[0,0].plot(self.X.flatten()[X_sort], true_lik[X_sort],'b-')
            self.global_plot_arr[0,0].plot(self.X.flatten()[X_sort], post_lik[X_sort],'g--')
            
            self.global_plot_arr[0,0].set_title("Density Plot")
            self.global_plot_arr[0,0].set_xlabel("X")
            self.global_plot_arr[0,0].set_ylabel("Density")            
            self.global_plot_arr[0,0].legend(["True","Final"])

            mu_counter = 0
            sigma_counter = 0
            for param in xrange(self.accept_chain.shape[1]):
                if param < self.K:
                    mu_counter += 1
                    self.global_plot_arr[1,0].plot(self.theta_hat_trace[:,param],'-',label="mu_"+str(mu_counter))                                
                else:
                    sigma_counter += 1 
                    self.global_plot_arr[1,0].plot(self.theta_hat_trace[:,param],'-.',label="sigma_"+str(sigma_counter))

            self.global_plot_arr[1,0].set_title("Traceplot")
            self.global_plot_arr[1,0].set_xlabel("Iteration")
            self.global_plot_arr[1,0].set_ylabel("Parameter")            
            self.global_plot_arr[1,0].legend(loc='lower center',ncol=2,bbox_to_anchor=(0., -.45, 1., .102),mode="expand")

            self.global_plot_arr[0,1].hist(self.accept_burnin[:,0])
            self.global_plot_arr[0,1].set_title("Traceplot mu_1")
            self.global_plot_arr[0,1].set_xlabel("mu_1")

            self.global_plot_arr[0,2].hist(self.accept_burnin[:,1])
            self.global_plot_arr[0,2].set_title("Traceplot mu_2")
            self.global_plot_arr[0,2].set_xlabel("mu_2")

            self.global_plot_arr[1,1].hist(self.accept_burnin[:,2])
            self.global_plot_arr[1,1].set_title("Traceplot sigma_1")
            self.global_plot_arr[1,1].set_xlabel("sigma_1")

            self.global_plot_arr[1,2].hist(self.accept_burnin[:,3])
            self.global_plot_arr[1,2].set_title("Traceplot sigma_2")
            self.global_plot_arr[1,2].set_xlabel("sigma_2")


            self.global_plot_f.subplots_adjust(hspace=.3, wspace=.3)
            global_plot_fname = fig_folder + "bimodal_final_MH_" + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".png"
            self.global_plot_f.savefig(os.path.abspath(global_plot_fname), dpi=600, format='png', bbox_inches='tight')
                                
    def sample(self):
        for it in xrange(self.iters):
            self.local_MH(it)
            print("P: %i\tIter: %i\tmu: %s\tsigma: %s\nLL: %.2f\tMH Acceptance: %.2f\tZ: %s" % (self.rank,it,self.mu_local_trace[it,:],self.sigma_local_trace[it,:],self.LL_local_trace[it],self.MH_prob_local[0]/self.MH_prob_local[1], self.Z_local_count))
                
        self.comm.barrier()        
        self.gather_chains()  

        if self.rank==0:            
            self.dp = BayesianGaussianMixture(n_components = min(self.burnin,100), max_iter=5000)
            self.dp.fit(self.theta)
            self.theta_hat_trace = np.empty((self.iters2,self.K*2))
            self.LL_hat_trace = np.empty((self.iters2,1))
            self.MH_prob_final = [0.,1.]
            for it in xrange(self.iters2):
                self.final_MH(it)
                print("Iter: %i\tTheta: %s\tLL: %.2f\tMH Acceptance: %.2f" % (it,self.theta_hat_trace[it],(self.LL_hat_trace[it]+self.LL_approx_mean),self.MH_prob_final[0]/self.MH_prob_final[1]))
#            self.LL_hat_trace *= self.LL_approx_sd
            self.LL_hat_trace -= self.LL_approx_mean
            self.plot_results()

        else:
            self.dp = None
            self.theta_hat_trace = None
            self.LL_hat_trace = None
            self.MH_prob_final = None
                        
    def local_MH(self,it):                           
        for k in xrange(self.K):
            data_k = self.X_local[np.where(self.Z_local == k)]
            X_bar = data_k.mean(axis=0)
            N_k = data_k.shape[0]
            mu_posterior = (self.prior_obs*self.prior_mu + N_k*X_bar)/(N_k + self.prior_obs)
            diff = data_k - X_bar
            prior_diff = X_bar - self.prior_mu
            nu_posterior = self.prior_obs + N_k

            a_post = self.prior_a + .5*N_k
            b_post = self.prior_b  + .5*np.sum(diff**2) + .5*((self.prior_obs * N_k)/nu_posterior)*(prior_diff**2)
            b_post = 1./(b_post)

            self.sigma_local_trace[it,k] = np.sqrt(1./np.random.gamma(a_post, b_post))
            self.mu_local_trace[it,k] = np.random.normal(mu_posterior, self.sigma_local_trace[it,k]/nu_posterior)            
#            self.LL_local_trace[it] += norm.logpdf(self.X_local[self.Z_local == k],self.mu_local_trace[it,k],self.sigma_local_trace[it,k])
#            self.LL_local_trace[it] += np.log(self.pi_local_trace[it,k])                
#            LL = norm.logpdf(self.X_local[self.Z_local == k],self.mu_local_trace[it,k],self.sigma_local_trace[it,k])
#            LL += np.log(self.pi_local_trace[it,k])
            

        if self.mu_local_trace[it,0] >= self.mu_local_trace[it,1]:
            resort = np.argsort(self.mu_local_trace[it,:])
            self.mu_local_trace[it,:] = self.mu_local_trace[it,resort]
            self.sigma_local_trace[it,:] = self.sigma_local_trace[it,resort]
            self.Z_local = (~(self.Z_local.astype(bool))).astype(int)
            self.Z_local_count = np.bincount(self.Z_local, minlength=self.K)
            
        # sample latent variables
        for i in xrange(self.N_p):
            self.Z_local_count[self.Z_local[i]] -= 1
            mixture_LL = norm.logpdf(self.X_local[i],self.mu_local_trace[it,:],self.sigma_local_trace[it,:])
            mixture_LL += np.log(self.Z_local_count + self.alpha)
            mixture_LL -= logsumexp(mixture_LL)
            mixture_LL = np.exp(mixture_LL)
            self.Z_local[i] = np.random.choice(int(self.K),p=mixture_LL)
            self.Z_local_count[self.Z_local[i]] += 1

        self.pi_local_trace[it] = np.random.dirichlet(self.Z_local_count + self.alpha)
        LL = np.sum([np.exp(np.log(self.pi_local_trace[it,k]) + norm.logpdf(self.X_local,self.mu_local_trace[it,k],self.sigma_local_trace[it,k])) for k in xrange(self.K)],axis=0)
        self.LL_local_trace[it] = np.log(LL).sum()

                    
    def final_MH(self,it):
        if it == 0:
            self.LL_hat_trace[it] = self.predict_LL(self.theta_mean_burnin, self.h_star)
            self.theta_hat_trace[it] = self.theta_mean_burnin
        else:
#            theta_mean = self.accept_burnin.mean(axis=0)
#            theta_sd = np.sqrt(self.accept_burnin.var(axis=0))
            theta_star = self.dp.sample()[0][0]
            old_theta = self.theta_hat_trace[it-1]
            proposal_LL = self.predict_LL(theta_star, self.h_star)
#            attempts = 0
#            proposal_LL = self.predict_LL(theta_star)                 
#            accept_prob = proposal_LL - self.LL_hat_trace[it-1] + self.dp.score_samples(old_theta) - self.dp.score_samples(theta_star)
            accept_prob = (proposal_LL+self.LL_approx_mean) - (self.LL_hat_trace[it-1]+self.LL_approx_mean)
            accept_prob += self.dp.score_samples(old_theta.reshape(1,-1)) - self.dp.score_samples(theta_star.reshape(1,-1))
            for k in xrange(self.K):
                accept_prob += np.array(norm.logpdf(theta_star[k], self.prior_mu, np.sqrt(self.prior_obs)*theta_star[k+2]) - norm.logpdf(old_theta[k], self.prior_mu, np.sqrt(self.prior_obs)*old_theta[k+2])).reshape(accept_prob.shape)
                accept_prob += np.array(gamma.logpdf(1./(theta_star[k+2]**2), a=self.prior_a, scale=1./self.prior_b) - gamma.logpdf(1./(old_theta[k+2]**2), a=self.prior_a, scale=1./self.prior_b)).reshape(accept_prob.shape)
#            pdb.set_trace() 
            u = np.log(np.random.uniform())
            self.MH_prob_final[1] +=1.
            if u < accept_prob.flatten():
                self.theta_hat_trace[it] = theta_star
                self.LL_hat_trace[it] = proposal_LL
                self.MH_prob_final[0] +=1.
            else:
                self.theta_hat_trace[it] = self.theta_hat_trace[it-1]
                self.LL_hat_trace[it] = self.LL_hat_trace[it-1]
        
    def predict_LL(self, theta_star, h):
#        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-theta_star,(self.theta-theta_star)**2))
#        W_i = rbf_kernel(theta_star.reshape(1,-1), self.theta, gamma=h) * np.eye(self.theta_n)
        theta_star_scale = self.scaler.transform(theta_star).reshape(1,-1)
#        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-theta_star,(self.theta-theta_star)**2))
#        W_i = h*rbf_kernel(theta_star.reshape(1,-1), self.theta, gamma=h) * np.eye(self.theta_n)
        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta_scale-theta_star_scale,(self.theta_scale-theta_star_scale)**2))
        W_i = (h/float(self.theta_n))*rbf_kernel(theta_star_scale.reshape(1,-1), self.theta_scale, gamma=h) * np.eye(self.theta_n)
#        W_i = h*rbf_kernel(theta_star_scale, self.theta_scale, gamma=h) * np.eye(self.theta_n)
        RW = np.dot(R_i.T, W_i)
        RW_R_i = np.dot(RW,R_i)        
        assert(np.linalg.cond(RW_R_i) < 1./sys.float_info.epsilon)
        RWR_inv = np.linalg.inv(RW_R_i)
        RWLL = np.dot(RW, self.LL_approx)
        beta_hat = np.dot(RWR_inv, RWLL)
        return(beta_hat[0])

    def local_reg_CV(self,h):
        """
        Objective function to optimize bandwidth parameter for LOOCV of local
        regression model, very slow!
        """
        CV = 0.        
        for i in xrange(self.theta_n):
            R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-self.theta[i],(self.theta-self.theta[i])**2))
            W_i = rbf_kernel(self.theta[i].reshape(1,-1), self.theta, gamma=np.exp(h)) * np.eye(self.theta_n)
            RW = np.dot(R_i.T, W_i)
            RWR_inv = np.linalg.inv( np.dot(RW,R_i))
            RWLL = np.dot(RW, self.LL_approx)
            beta_hat = np.dot(RWR_inv, RWLL)
            H_ii = np.dot(np.dot(R_i,RWR_inv),RW)[i,i]
            Y_hat = beta_hat[0]
            CV += ((self.LL_approx[i] - Y_hat) / (1.-H_ii))**2
        return(CV)

    def local_reg_risk(self,h):
        """
        Objective function to optimize bandwidth parameter for risk estimater of
        local regression
        """
        h = np.exp(h)        
        W = (h/float(self.theta_n))*rbf_kernel(self.theta_scale,gamma=h)
        risk = np.mean((self.LL_approx - np.dot(W,self.LL_approx))**2)/float(self.theta_n)
        sigma_hat = np.mean((self.LL_approx[1:]-self.LL_approx[:-1])**2)
        sigma_hat /= 2.
        risk -= sigma_hat
        risk += 2.*sigma_hat*h/float(self.theta_n)
        return(risk)
        

if __name__ == '__main__':
    synthetic_data = loadmat("../data/bimodal.mat")
    pmc = parallelMCMC_bimodal(X=synthetic_data['X'],Z=synthetic_data['Z'],alpha=1.,
                          mu=synthetic_data['mu'][0],sigma=synthetic_data['sigma'][0],
                          pi=synthetic_data['pi'][0])
    pmc.sample()