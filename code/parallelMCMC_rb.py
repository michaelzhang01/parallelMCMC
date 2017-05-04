# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 09:28:17 2016

Parallel MCMC--Rare Bernoulli Example

@author: Michael Zhang
"""

import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from mpi4py import MPI
from scipy.io import loadmat, savemat
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import beta
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

class parallelMCMC_rb(object):
    
    def __init__(self, X,theta=None,a=None, b=None, 
                 iters=1000, iters2=20000):
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
        self.prior_a = a
        self.prior_b = b
        self.true_theta = theta
        self.dim = 1.
        self.scaling = ((2.4)**2)/self.dim
        self.iters = iters 
        self.burnin = self.iters // 2                
        self.iters2 = iters2
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()        
        self.rank = self.comm.Get_rank()        
        if self.comm.rank == 0:
            self.X = X.astype(float)
        else:
            self.X = None
            self.LL_eval = None
            self.interpolate_LL = None
            self.LL_grid = None

        self.X = self.comm.bcast(self.X)
        self.X_sum = self.X.sum()
        self.N,self.D = self.X.shape
        
        self.part_size_X = tuple([j * self.D for j in self.partition_tuple()])
        self.data_displace_X = self.displacement(self.part_size_X)
        self.X_local = np.zeros(self.partition_tuple()[self.rank] * self.D)        
        self.comm.Scatterv([self.X, self.part_size_X, self.data_displace_X, MPI.DOUBLE], self.X_local)                    

        self.X_local = self.X_local
        self.X_local = self.X_local.reshape((-1, self.D))     
        self.N_p,self.D_p = self.X_local.shape
        assert(self.D_p == self.D)
        assert(self.N_p > 0)
        self.theta_local_trace = np.empty(self.iters)
        self.theta_local_all_trace = np.empty((self.iters,1))        
        
        self.LL_local_trace = np.empty(self.iters)
        self.LL_local_all_trace = np.empty((self.iters,1))

        self.MH_prob_local = [0.,1.]
        
        if self.rank == 0:
            print("Initialization Complete")
        else:
            self.X = None
            
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
        self.accept_chain = np.array(self.comm.allgather(self.theta_local_trace)).reshape(-1,1)
        self.accept_burnin = np.array(self.comm.allgather(self.theta_local_trace[self.burnin:])).reshape(-1,1)

        accept_n, accept_d = self.accept_chain.shape
        self.theta = np.array(self.comm.allgather(self.theta_local_all_trace[self.burnin:])).reshape(-1,1)
        self.theta_n, self.theta_d = self.theta.shape
        local_LL_eval = np.array([np.log((th*(self.X_local==1) + (1.-th)*(self.X_local==0))).sum() for th in self.theta.flatten()])
        local_LL_eval = np.array([np.log((th*(self.X_local==1) + (1.-th)*(self.X_local==0))).sum() for th in self.theta.flatten()])
        local_accept_LL = np.array([np.log((th*(self.X_local==1) + (1.-th)*(self.X_local==0))).sum() for th in self.accept_chain.flatten()])
        self.LL_approx = self.comm.reduce(local_LL_eval)
        self.LL_accept_approx = self.comm.reduce(local_accept_LL)
#        local_LL_eval = np.array([np.log((th*(self.X_local==1) + (1.-th)*(self.X_local==0))).sum() for th in self.theta.flatten()])
#        local_accept_LL = np.array([np.log((th*(self.X_local==1) + (1.-th)*(self.X_local==0))).sum() for th in self.accept_chain.flatten()])
#        self.LL_approx = self.comm.gather(local_LL_eval)
#        self.LL_accept_approx = self.comm.gather(local_accept_LL)

#        pdb.set_trace()
        if self.rank==0:
            self.theta_mean_burnin = self.accept_burnin.mean()
            self.theta_cov_burnin = np.var(self.accept_burnin)
#            self.LL_approx = self.P*np.median(self.LL_approx,axis=0)
#            self.LL_accept_approx = self.P*np.median(self.LL_accept_approx,axis=0)
            self.LL_approx_sd = np.sqrt(self.LL_approx.var())
            self.LL_approx_mean = self.LL_approx.mean()
            self.LL_approx -= self.LL_approx_mean
            self.scaler = StandardScaler()
            self.theta_scale = self.scaler.fit_transform(self.theta)
#            self.h_star = np.exp(minimize_scalar(self.local_reg_risk).x)
            self.h_star = np.exp(minimize(self.local_reg_risk,[0]).x[0])
#            self.h_star = np.exp(minimize_scalar(self.local_reg_CV).x)
#            self.LL_approx /= self.LL_approx_sd
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

            self.local_plot_f, self.local_plot_arr = plt.subplots(1,3, figsize=(15.75,4))
            arange_grid = np.arange(0.,.02,.001)
            true_LL_grid = np.array([(self.X_sum)*np.log(th) + (self.N - self.X_sum)*np.log(1.- th) + beta.logpdf(th, self.prior_a,self.prior_b) for th in arange_grid])
            true_posterior_1 = np.random.beta(self.X_sum+self.prior_a, self.N+self.prior_b-self.X_sum, self.accept_burnin.shape[0])
            true_posterior_2 = np.random.beta(self.X_sum+self.prior_a, self.N+self.prior_b-self.X_sum, self.iters2)
            local_posterior = self.LL_accept_approx + beta.logpdf(self.accept_chain.flatten(),self.prior_a,self.prior_b)
            self.local_plot_arr[0].plot(self.accept_chain, local_posterior, 'x')
            self.local_plot_arr[0].plot(arange_grid, true_LL_grid,'-')
#            y_lim_LL = int(self.local_plot_arr[0].get_ylim()[0])
#            self.local_plot_arr[0].plot([self.true_theta]*y_lim_LL, xrange(y_lim_LL))
            self.local_plot_arr[0].legend(['MCMC', 'Post.'])
            self.local_plot_arr[0].set_title("Likelihood Plot")            
            self.local_plot_arr[0].set_xlabel("Theta")
            self.local_plot_arr[0].set_ylabel("Log Likelihood")
 
            for tr in xrange(0,self.P*self.iters, self.iters):
                self.local_plot_arr[1].plot(xrange(self.iters),self.accept_chain[tr:tr+self.iters],'o')
                            
            self.local_plot_arr[1].plot([self.true_theta]*int(self.local_plot_arr[1].get_xlim()[1]),'r-',linewidth=2)
            self.local_plot_arr[1].set_title("Traceplot Theta")
            self.local_plot_arr[1].set_xlabel("Iterations")
            self.local_plot_arr[1].set_ylabel("Theta")
            
            self.local_plot_arr[2].hist(self.accept_burnin,alpha=.5)
            self.local_plot_arr[2].hist(true_posterior_1,alpha=.5)
            self.local_plot_arr[2].legend(['MCMC', 'Post.'],loc=0)
            self.local_plot_arr[2].vlines(self.true_theta, ymin=self.local_plot_arr[2].get_ylim()[0],ymax=self.local_plot_arr[2].get_ylim()[1], colors='r', linewidth=2)
            self.local_plot_arr[2].set_title("Histogram Theta")
            self.local_plot_arr[2].set_xlabel("Theta")            
            self.local_plot_f.subplots_adjust(wspace=.3)
            local_plot_fname = fig_folder + "rb_subset_MH_"  + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".png"
            self.local_plot_f.savefig(os.path.abspath(local_plot_fname), dpi=600, format='png', bbox_inches='tight')
            
            self.global_plot_f, self.global_plot_arr = plt.subplots(1,3, figsize=(15.75,4))
            global_posterior = self.LL_hat_trace.flatten() +  beta.logpdf(self.theta_hat_trace.flatten(),self.prior_a,self.prior_b)
            self.global_plot_arr[0].plot(self.theta_hat_trace, global_posterior,'x')
            self.global_plot_arr[0].plot(arange_grid, true_LL_grid,'-')
            self.global_plot_arr[0].legend(['MCMC', 'Post.'],loc=0)

#            y_lim_LL = int(self.global_plot_arr[0].get_ylim()[0])
            
#            self.global_plot_arr[0].plot([self.true_theta]*y_lim_LL, xrange(y_lim_LL))
            self.global_plot_arr[0].set_title("Likelihood Plot")
            self.global_plot_arr[0].set_xlabel("Theta")
            self.global_plot_arr[0].set_ylabel("Log Likelihood")
 
            self.global_plot_arr[1].plot(self.theta_hat_trace,'b.')
            
            self.global_plot_arr[1].plot([self.true_theta]*int(self.global_plot_arr[1].get_xlim()[1]),'r-',linewidth=2)
            self.global_plot_arr[1].set_title("Traceplot Theta")
            self.global_plot_arr[1].set_xlabel("Iterations")
            self.global_plot_arr[1].set_ylabel("Theta")
            
            self.global_plot_arr[2].hist(self.theta_hat_trace,alpha=.5)
            self.global_plot_arr[2].hist(true_posterior_2,alpha=.5)
            self.global_plot_arr[2].legend(['MCMC', 'Post.'],loc=0)
            self.global_plot_arr[2].set_title("Histogram Theta")
            self.global_plot_arr[2].vlines(self.true_theta, ymin=self.global_plot_arr[2].get_ylim()[0],ymax=self.global_plot_arr[2].get_ylim()[1], colors='r', linewidth=2)
            self.global_plot_arr[2].set_xlabel("Theta")            
            self.global_plot_f.subplots_adjust(wspace=.3)

            global_plot_fname = fig_folder + "rb_final_MH_" + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".png"
            self.global_plot_f.savefig(os.path.abspath(global_plot_fname), dpi=600, format='png', bbox_inches='tight')
            save_dict = {'local_chain':self.accept_burnin, 'global_chain':self.theta_hat_trace}
            dict_fname = fig_folder + "rb_mat_" + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".mat"
            savemat(dict_fname,save_dict)

    def sample(self):
        for it in xrange(self.iters):
            self.local_MH(it)
            print("P: %i\tIter: %i\ttheta: %.3f\tLL: %.2f\tMH Acceptance: %.2f" % (self.rank,it,self.theta_local_trace[it],self.LL_local_trace[it],self.MH_prob_local[0]/self.MH_prob_local[1]))
                
        self.comm.barrier()        
#        self.GR()
        self.gather_chains()  

        if self.rank==0:            
            self.dp = BayesianGaussianMixture(n_components = 100, max_iter=5000)
            self.dp.fit(self.theta)
            self.theta_hat_trace = np.empty((self.iters2,1))
            self.LL_hat_trace = np.empty((self.iters2,1))
            self.MH_prob_final = [0.,1.]
            for it in xrange(self.iters2):
                self.final_MH(it)
                print("Iter: %i\ttheta: %.3f\tLL: %.2f\tMH Acceptance: %.2f" % (it,self.theta_hat_trace[it],(self.LL_hat_trace[it]+self.LL_approx_mean),self.MH_prob_final[0]/self.MH_prob_final[1]))
#            self.LL_hat_trace *= self.LL_approx_sd
            self.LL_hat_trace += self.LL_approx_mean
            self.plot_results()

        else:
            self.dp = None
            self.theta_hat_trace = None
            self.LL_hat_trace = None
            self.MH_prob_final = None
        
                
    def local_MH(self,it):
        if it == 0:
            self.theta_local_trace[it] = np.random.beta(self.prior_a + self.X_local.sum(),self.prior_b + self.N_p - self.X_local.sum())
            self.LL_local_trace[it] = np.log(1.-self.theta_local_trace[it]*(self.X_local==0) + (self.theta_local_trace[it])*(self.X_local==1)).sum()
        else:
            self.theta_local_all_trace[it] = np.random.beta(self.prior_a + self.X_local.sum(),self.prior_b + self.N_p - self.X_local.sum())
            proposal_LL = (np.log(self.theta_local_all_trace[it]) * self.X_local.sum()) + (np.log(1.-self.theta_local_all_trace[it]) * (self.N_p - self.X_local.sum()))
#            accept_prob = proposal_LL - self.LL_local_trace[it-1] + beta.logpdf(self.theta_local_all_trace[it],self.prior_a,self.prior_b) - beta.logpdf(self.theta_local_trace[it-1],self.prior_a,self.prior_b) - beta.logpdf(self.theta_local_all_trace[it],self.prior_a + self.X_local.sum(),self.prior_b + self.N_p - self.X_local.sum()) + beta.logpdf(self.theta_local_trace[it-1],self.prior_a + self.X_local.sum(),self.prior_b + self.N_p - self.X_local.sum())
#            u = np.log(np.random.uniform())
            self.MH_prob_local[1] += 1.
            self.theta_local_trace[it] = np.copy(self.theta_local_all_trace[it])
            self.LL_local_trace[it] = proposal_LL
            self.MH_prob_local[0] +=1.

#            if u < accept_prob:
#                self.theta_local_trace[it] = np.copy(self.theta_local_all_trace[it])
#                self.LL_local_trace[it] = proposal_LL
#                self.MH_prob_local[0] +=1.
#            else:
#                self.theta_local_trace[it] = np.copy(self.theta_local_trace[it-1])
#                self.LL_local_trace[it] = np.copy(self.LL_local_trace[it-1])
        
    def final_MH(self,it):
        if it == 0:
            self.LL_hat_trace[it] = self.predict_LL(np.array(self.theta_mean_burnin).reshape(-1,1),self.h_star)
            self.theta_hat_trace[it] = self.theta_mean_burnin
        else:
            theta_star = np.abs(self.dp.sample()[0])
            while theta_star.flatten() > 1.:
                theta_star = np.abs(self.dp.sample()[0])
            old_theta = self.theta_hat_trace[it-1]
            proposal_LL = self.predict_LL(theta_star,self.h_star)                 
#            accept_prob = (self.LL_approx_sd*proposal_LL+self.LL_approx_mean) - (self.LL_approx_sd*self.LL_hat_trace[it-1]+self.LL_approx_mean) 
            accept_prob = (proposal_LL+self.LL_approx_mean) - (self.LL_hat_trace[it-1]+self.LL_approx_mean) 
            accept_prob += self.dp.score_samples(old_theta.reshape(-1,1)) - self.dp.score_samples(theta_star.reshape(-1,1))# + beta.logpdf(theta_star,self.prior_a,self.prior_b).reshape(-1,1) - beta.logpdf(old_theta,self.prior_a,self.prior_b).reshape(-1,1)
            accept_prob += np.array(beta.logpdf(theta_star,self.prior_a,self.prior_b) - beta.logpdf(old_theta,self.prior_a,self.prior_b)).reshape(accept_prob.shape)
            
            u = np.log(np.random.uniform())
            self.MH_prob_final[1] +=1.

            if (u < accept_prob.flatten()):
                self.theta_hat_trace[it] = theta_star
                self.LL_hat_trace[it] = proposal_LL.flatten()
                self.MH_prob_final[0] +=1.
            else:
                self.theta_hat_trace[it] = self.theta_hat_trace[it-1]
                self.LL_hat_trace[it] = self.LL_hat_trace[it-1]
            if np.isnan(self.LL_hat_trace[it]):
                print(proposal_LL)
                print(self.LL_approx)
        
    def predict_LL(self, theta_star, h=.0005):
        theta_star_scale = self.scaler.transform(theta_star)
#        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-theta_star,(self.theta-theta_star)**2))
#        W_i = h*rbf_kernel(theta_star.reshape(1,-1), self.theta.reshape(-1,1), gamma=h) * np.eye(self.theta_n)
        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta_scale-theta_star_scale,(self.theta_scale-theta_star_scale)**2))
        W_i = (h/float(self.theta_n))*rbf_kernel(theta_star_scale.reshape(1,-1), self.theta_scale.reshape(-1,1), gamma=h) * np.eye(self.theta_n)
        RW = np.dot(R_i.T, W_i)
        RWR_inv = np.linalg.inv( np.dot(RW,R_i))
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
    synthetic_data = loadmat("../data/rare_bernoulli.mat")
    N = synthetic_data['N'][0][0]
    resample = np.random.choice(a=N,size=N,replace=False)
    pmc = parallelMCMC_rb(X=synthetic_data['X'][resample,:],theta=synthetic_data['theta'][0][0],
                          a=synthetic_data['a'][0][0], b=synthetic_data['b'][0][0])
    pmc.sample()