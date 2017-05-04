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
#import pdb
from mpi4py import MPI
from scipy.io import loadmat
from scipy.interpolate import Rbf
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

class parallelMCMC(object):
    
    def __init__(self, X,Y,Z,a=None, b=None, iters=1000, iters2=5000, 
                 prior_var = 1000, prior_mean = [0.,0.], proposal_tune = .5,
                 random_walk = True):
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
        self.true_a = a
        self.true_b = b
        self.dim = 2
        self.scaling = ((2.4)**2)/self.dim
#        self.proposal_tune_2 = proposal_tune_2
        self.random_walk = bool(random_walk)
        self.prior_var = prior_var
        self.prior_mean = np.array(prior_mean)
        self.proposal_tune = proposal_tune
        self.iters = iters
        self.burnin = self.iters // 2                
        self.iters2 = iters2
        self.comm = MPI.COMM_WORLD
#        self.today = datetime.datetime.today()        
        self.P = self.comm.Get_size()        
        self.rank = self.comm.Get_rank()        
        if self.comm.rank == 0:
            self.X = X
            self.Y = Y.astype(float)
            self.Z = Z.astype(float)
        else:
            self.X = None
            self.Y = None
            self.Z = None
            self.a_grid = None
            self.b_grid = None
            self.a_mesh = None
            self.b_mesh = None
            self.LL_eval = None
            self.interpolate_LL = None
            self.LL_grid = None

        self.X = self.comm.bcast(self.X)
        self.Y = self.comm.bcast(self.Y)        
        self.Z = self.comm.bcast(self.Z) 
        self.N,self.D = self.X.shape
        
        self.part_size_X = tuple([j * self.D for j in self.partition_tuple()])
        self.part_size_Y = tuple([j for j in self.partition_tuple()])
        self.part_size_Z = tuple([j for j in self.partition_tuple()])
        self.data_displace_X = self.displacement(self.part_size_X)
        self.data_displace_Y = self.displacement(self.part_size_Y)        
        self.data_displace_Z = self.displacement(self.part_size_Y)        
        self.X_local = np.zeros(self.partition_tuple()[self.rank] * self.D)        
        self.Y_local = np.zeros((self.partition_tuple()[self.rank]))
        self.Z_local = np.zeros((self.partition_tuple()[self.rank]))
        self.comm.Scatterv([self.Y, self.part_size_Y, self.data_displace_Y, MPI.DOUBLE], self.Y_local)                
        self.comm.Scatterv([self.X, self.part_size_X, self.data_displace_X, MPI.DOUBLE], self.X_local)                    
        self.comm.Scatterv([self.Z, self.part_size_Z, self.data_displace_Z, MPI.DOUBLE], self.Z_local)

        self.X_local = self.X_local.reshape((-1, self.D))     
        self.N_p,self.D_p = self.X_local.shape
        assert(self.D_p == self.D)
        assert(self.N_p > 0)
        self.Y_local = self.Y_local.reshape(self.N_p, -1).astype(int)
        self.Z_local = self.Z_local.reshape(self.N_p, -1).astype(int)
        
        self.a_local_trace = np.empty(self.iters) # holds all MH acceptances
        self.b_local_trace = np.empty(self.iters)
        self.LL_local_trace = np.empty(self.iters)
        self.a_local_all_trace = np.empty((self.iters,1)) # holds all MH proposals
        self.b_local_all_trace = np.empty((self.iters,1))
        self.LL_local_all_trace = np.empty((self.iters,1))

        self.MH_prob_local = [0.,1.]
        
        if self.rank == 0:
            print("Initialization Complete")
        else:
            self.X = None
            self.Y = None
            self.Z = None
            
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
        self.accept_chain = np.hstack((np.array(self.comm.allgather(self.a_local_trace)).reshape(-1,1),
                                np.array(self.comm.allgather(self.b_local_trace)).reshape(-1,1)))
        self.accept_burnin = np.hstack((np.array(self.comm.allgather(self.a_local_trace[self.burnin:])).reshape(-1,1),
                                        np.array(self.comm.allgather(self.b_local_trace[self.burnin:])).reshape(-1,1)))        

        accept_n, accept_d = self.accept_chain.shape
        self.theta = np.hstack((np.array(self.comm.allgather(self.a_local_all_trace[self.burnin:])).reshape(-1,1),
                                np.array(self.comm.allgather(self.b_local_all_trace[self.burnin:])).reshape(-1,1)))
        self.theta_n, self.theta_d = self.theta.shape
        local_LL_eval = np.array([self.local_LL(self.theta[i,0], self.theta[i,1]) for i in xrange(self.theta_n)])
        local_accept_LL = np.array([self.local_LL(self.accept_chain[i,0], self.accept_chain[i,1]) for i in xrange(accept_n)])
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
            self.LL_approx /= self.LL_approx_sd
        else:            
            self.theta = None
            self.theta_mean_burnin = None
            self.LL_approx_sd = None
            self.LL_approx_mean = None
            self.LL_gather = None
            self.LL_approx = None
            self.LL_accept_approx = None

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

            a_lin_local = np.linspace(self.accept_chain[:,0].min()-10, 
                                self.accept_chain[:,0].max()+10, grid_size) #
            b_lin_local = np.linspace(self.accept_chain[:,1].min()-10, 
                                self.accept_chain[:,1].max()+10, grid_size) #

            iter_prod = itertools.product(a_lin_local, b_lin_local)
            self.a_mesh_local, self.b_mesh_local = np.meshgrid(a_lin_local,b_lin_local)
            self.a_grid_local, self.b_grid_local,self.LL_eval_local = np.array([(a0,b0,self.full_LL(a0,b0)) for a0,b0 in iter_prod]).T
            self.interpolate_LL_local = Rbf(self.a_grid_local, self.b_grid_local, self.LL_eval_local, function='linear')
            self.LL_grid_local = self.interpolate_LL_local(self.a_mesh_local, self.b_mesh_local)

            self.local_plot_f, self.local_plot_arr = plt.subplots(3,2, figsize=(12,10.5))
            self.local_plot_arr[0,0].contourf(self.LL_grid_local, vmin=min((self.LL_eval_local.min(),self.LL_approx.min())),
                                        vmax=max(self.LL_eval_local.max(),self.LL_approx.max()),
                                        extent=[self.a_grid_local.min(),self.a_grid_local.max(),
                                                self.b_grid_local.min(),self.b_grid_local.max()],origin='lower')#,levels=levels)
            self.local_plot_arr[0,0].scatter(self.accept_chain[:,0],self.accept_chain[:,1],
                                c=self.LL_accept_approx,vmin=min((self.LL_eval_local.min(),self.LL_approx.min())),
                                vmax=max(self.LL_eval_local.max(),self.LL_approx.max()),linewidths = .2 ,marker='.',s = 100)
            self.local_plot_arr[0,0].scatter(self.true_a,self.true_b,marker='x', s= 200, c='k')            
            self.local_plot_arr[0,0].set_title("Joint Likelihood")
            self.local_plot_arr[0,0].set_xlabel("a")
            self.local_plot_arr[0,0].set_ylabel("b")
 
            self.local_plot_arr[0,1].hist2d(self.accept_burnin[:,0],self.accept_burnin[:,1], normed=True)            
            self.local_plot_arr[0,1].set_xlabel("a")
            self.local_plot_arr[0,1].set_ylabel("b")
            self.local_plot_arr[0,1].set_title("Joint Histogram")

            for tr in xrange(0,self.P*self.iters, self.iters):
                self.local_plot_arr[1,0].plot(self.accept_chain[tr:tr+self.iters,0],'b.')
                self.local_plot_arr[1,1].plot(self.accept_chain[tr:tr+self.iters,1],'b.')
                            
            self.local_plot_arr[1,0].plot([self.true_a]*int(self.local_plot_arr[1,0].get_xlim()[1]),'r-',linewidth=2)
            self.local_plot_arr[1,0].set_title("Traceplot a")
            self.local_plot_arr[1,0].set_xlabel("Iterations")
            self.local_plot_arr[1,0].set_ylabel("a")
            
            self.local_plot_arr[1,1].plot([self.true_b]*int(self.local_plot_arr[1,1].get_xlim()[1]),'r-',linewidth=2)
            self.local_plot_arr[1,1].set_title("Traceplot b")
            self.local_plot_arr[1,1].set_xlabel("Iterations")
            self.local_plot_arr[1,1].set_ylabel("b")

            self.local_plot_arr[2,0].hist(self.accept_burnin[:,0])
            self.local_plot_arr[2,0].set_title("Histogram a")
            self.local_plot_arr[2,0].vlines(self.true_a, ymin=self.local_plot_arr[2,0].get_ylim()[0],ymax=self.local_plot_arr[2,0].get_ylim()[1], colors='r', linewidth=2)
            self.local_plot_arr[2,0].set_xlabel("a")            
            
            self.local_plot_arr[2,1].hist(self.accept_burnin[:,1])
            self.local_plot_arr[2,1].set_title("Histogram b")
            self.local_plot_arr[2,1].vlines(self.true_b, ymin=self.local_plot_arr[2,1].get_ylim()[0],ymax=self.local_plot_arr[2,1].get_ylim()[1], colors='r', linewidth=2)
            self.local_plot_arr[2,1].set_xlabel("b")            
            self.local_plot_f.subplots_adjust(hspace=.5, wspace=.3)

            local_plot_fname = fig_folder + "subset_MH_"  + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".png"
            self.local_plot_f.savefig(os.path.abspath(local_plot_fname), dpi=600, format='png', bbox_inches='tight')

            
            a_lin_global = np.linspace(self.a_hat_trace.min()-10, 
                                       self.a_hat_trace.max()+10, grid_size) #
            b_lin_global = np.linspace(self.b_hat_trace.min()-10, 
                                       self.b_hat_trace.max()+10, grid_size) #

            iter_prod = itertools.product(a_lin_global, b_lin_global)
            self.a_mesh_global, self.b_mesh_global = np.meshgrid(a_lin_global,b_lin_global)
            self.a_grid_global, self.b_grid_global,self.LL_eval_global = np.array([(a0,b0,self.full_LL(a0,b0)) for a0,b0 in iter_prod]).T
            self.LL_eval_global[np.isneginf(self.LL_eval_global)] = -1e100

            self.interpolate_LL_global = Rbf(self.a_grid_global, self.b_grid_global, self.LL_eval_global, function='linear')
            self.LL_grid_global = self.interpolate_LL_global(self.a_mesh_global, self.b_mesh_global)
            self.global_plot_f, self.global_plot_arr = plt.subplots(3,2, figsize=(12,10.5))
            self.global_plot_arr[0,0].contourf(self.LL_grid_global, vmin=min((self.LL_eval_global.min(),self.LL_approx.min())),
                                        vmax=max(self.LL_eval_global.max(),self.LL_approx.max()),
                                        extent=[self.a_grid_global.min(),self.a_grid_global.max(),
                                                self.b_grid_global.min(),self.b_grid_global.max()],origin='lower')#,levels=levels)
            self.global_plot_arr[0,0].scatter(self.a_hat_trace,self.b_hat_trace,
                                    c=self.LL_hat_trace,vmin=min((self.LL_eval_global.min(),self.LL_approx.min(),self.LL_hat_trace.min())),
                                    vmax=max(self.LL_eval_global.max(),self.LL_approx.max(),self.LL_hat_trace.max()),linewidths = .2,marker='.',s = 100)
            self.global_plot_arr[0,0].scatter(self.true_a,self.true_b,marker='x', s= 200, c='k')
            self.global_plot_arr[0,0].set_title("Joint Likelihood")
            self.global_plot_arr[0,0].set_xlabel("a")
            self.global_plot_arr[0,0].set_ylabel("b")

            self.global_plot_arr[0,1].hist2d(self.a_hat_trace.flatten(),self.b_hat_trace.flatten(), normed=True)            
            self.global_plot_arr[0,1].set_xlabel("a")
            self.global_plot_arr[0,1].set_ylabel("b")
            self.global_plot_arr[0,1].set_title("Joint Histogram")

            self.global_plot_arr[1,0].plot(self.a_hat_trace,'b.')
            self.global_plot_arr[1,0].plot([self.true_a]*self.iters2,'r-',linewidth=2)
            self.global_plot_arr[1,0].set_title("Traceplot a")
            self.global_plot_arr[1,0].set_xlabel("Iterations")
            self.global_plot_arr[1,0].set_ylabel("a")

            self.global_plot_arr[1,1].plot(self.b_hat_trace,'b.')
            self.global_plot_arr[1,1].plot([self.true_b]*self.iters2,'r-',linewidth=2)
            self.global_plot_arr[1,1].set_title("Traceplot b")
            self.global_plot_arr[1,1].set_xlabel("Iterations")
            self.global_plot_arr[1,1].set_ylabel("b")
            
            self.global_plot_arr[2,0].hist(self.a_hat_trace)
            self.global_plot_arr[2,0].set_title("Histogram a")
            self.global_plot_arr[2,0].vlines(self.true_a, ymin=self.global_plot_arr[2,0].get_ylim()[0],ymax=self.global_plot_arr[2,0].get_ylim()[1], colors='r', linewidth=2)
            self.global_plot_arr[2,0].set_xlabel("a")            

            self.global_plot_arr[2,1].hist(self.b_hat_trace)
            self.global_plot_arr[2,1].set_title("Histogram b")
            self.global_plot_arr[2,1].vlines(self.true_b, ymin=self.global_plot_arr[2,0].get_ylim()[0],ymax=self.global_plot_arr[2,0].get_ylim()[1], colors='r', linewidth=2)
            self.global_plot_arr[2,1].set_xlabel("b")            

            self.global_plot_f.subplots_adjust(hspace=.5, wspace=.3)
            global_plot_fname = fig_folder + "final_MH_" + str(self.P) + "_" + fname_today.strftime("%Y-%m-%d-%f") + ".png"
            self.global_plot_f.savefig(os.path.abspath(global_plot_fname), dpi=600, format='png', bbox_inches='tight')
            
            self.LL_plot_f, self.LL_plot_arr = plt.subplots(1,2, figsize=(12,3.5))

            iter_prod = itertools.product(a_lin_global, b_lin_global)
            self.LL_eval_approx = np.array([self.predict_LL(np.array([a0,b0])) for a0,b0 in iter_prod])
            self.interpolate_LL_approx = Rbf(self.a_grid_global, self.b_grid_global, self.LL_eval_approx, function='linear')
            self.LL_grid_approx = self.interpolate_LL_global(self.a_mesh_global, self.b_mesh_global)


            self.LL_plot_arr[0].contourf(self.LL_grid_global, vmin=min((self.LL_eval_global.min(),self.LL_eval_approx.min())),
                                        vmax=max(self.LL_eval_global.max(),self.LL_eval_approx.max()),
                                        extent=[self.a_grid_global.min(),self.a_grid_global.max(),
                                                self.b_grid_global.min(),self.b_grid_global.max()],origin='lower')#,levels=levels)
            self.LL_plot_arr[0].set_title("True Joint Likelihood")
            self.LL_plot_arr[0].set_xlabel("a")
            self.LL_plot_arr[0].set_ylabel("b")


            plot_1 = self.LL_plot_arr[1].contourf(self.LL_grid_approx, vmin=min((self.LL_eval_global.min(),self.LL_eval_approx.min())),
                                        vmax=max(self.LL_eval_global.max(),self.LL_eval_approx.max()),
                                        extent=[self.a_grid_global.min(),self.a_grid_global.max(),
                                                self.b_grid_global.min(),self.b_grid_global.max()],origin='lower')#,levels=levels)
            self.LL_plot_arr[1].set_title("Local Polynomial Approximate Joint Likelihood")
            self.LL_plot_arr[1].set_xlabel("a")
            self.LL_plot_arr[1].set_ylabel("b")
            self.LL_plot_f.colorbar(plot_1,ax=self.LL_plot_arr[1],pad=.01)

            LL_plot_fname = fig_folder +"LL_contour_" + str(self.P) + "_"  +fname_today.strftime("%Y-%m-%d-%f") + ".png"
            self.LL_plot_f.savefig(os.path.abspath(LL_plot_fname), dpi=600, format='png', bbox_inches='tight')
#            save_dict = {'local_chain':self.accept_burnin,'final_chain:}

#            plt.show()
                    
    def sample(self):
        for it in xrange(self.iters):
#            self.local_MH(it)
            self.local_ess(it)
            print("P: %i\tIter: %i\ta: %.2f\tb: %.2f\tLL: %.2f\tMH Acceptance: %.2f" % (self.rank,it,self.a_local_trace[it],self.b_local_trace[it],self.LL_local_trace[it],self.MH_prob_local[0]/self.MH_prob_local[1]))
                
        self.comm.barrier()        
#        self.GR()
        self.gather_chains()  

        if self.rank==0:            
            if self.random_walk:
                self.dp = None
            else:                
                self.dp = BayesianGaussianMixture(n_components = 100, max_iter=5000)
                self.dp.fit(self.theta)
            self.a_hat_trace = np.empty((self.iters2,1))
            self.b_hat_trace = np.empty((self.iters2,1))
            self.LL_hat_trace = np.empty((self.iters2,1))
            self.MH_prob_final = [0.,1.]
            self.scaler = StandardScaler()
            self.theta_scale = self.scaler.fit_transform(self.theta)
#            self.h_star = np.exp(minimize_scalar(self.local_reg_risk).x)
            self.h_star = np.exp(minimize(self.local_reg_risk,[0]).x[0])
            
            for it in xrange(self.iters2):
                self.final_MH(it)
#                print("Iter: %i\ta: %.2f\tb: %.2f\tLL: %.2f\tMH Acceptance: %.2f" % (it,self.a_hat_trace[it],self.b_hat_trace[it],(self.LL_approx_sd*self.LL_hat_trace[it]+self.LL_approx_mean),self.MH_prob_final[0]/self.MH_prob_final[1]))
                print("Iter: %i\ta: %.2f\tb: %.2f\tLL: %.2f\tMH Acceptance: %.2f" % (it,self.a_hat_trace[it],self.b_hat_trace[it],(self.LL_hat_trace[it]+self.LL_approx_mean),self.MH_prob_final[0]/self.MH_prob_final[1]))

#            self.LL_hat_trace *= self.LL_approx_sd
            self.LL_hat_trace += self.LL_approx_mean
            self.plot_results()

        else:
            self.dp = None
            self.a_hat_trace = None
            self.b_hat_trace = None
            self.LL_hat_trace = None
            self.MH_prob_final = None
        
                
    def local_MH(self,it):
        if it == 0:
            self.a_local_trace[it],self.b_local_trace[it] = np.random.normal(loc = self.prior_mean, scale = np.sqrt(self.prior_var), size=2)
            self.LL_local_trace[it] = self.local_LL(self.a_local_trace[it], 
                                                    self.b_local_trace[it])
            self.a_local_all_trace[it] = np.copy(self.a_local_trace[it])
            self.b_local_all_trace[it] = np.copy(self.b_local_trace[it])
#        elif it < self.burnin:
#            proposal_a = np.random.normal(loc = self.a_local_trace[it-1],
#                                          scale =self.proposal_tune)
#            proposal_b = np.random.normal(loc = self.b_local_trace[it-1],
#                                          scale =self.proposal_tune)
#            proposal_LL = self.local_LL(proposal_a, proposal_b)
#            accept_prob = proposal_LL - self.LL_local_trace[it-1]
#            u = np.log(np.random.uniform())
#            self.MH_prob_local[1] += 1.
#            self.a_local_all_trace[it] = proposal_a
#            self.b_local_all_trace[it] = proposal_b
#
#            if u < accept_prob:
#                self.a_local_trace[it] = proposal_a
#                self.b_local_trace[it] = proposal_b
#                self.LL_local_trace[it] = proposal_LL
#                self.MH_prob_local[0] +=1.
#            else:
#                self.a_local_trace[it] = np.copy(self.a_local_trace[it-1])
#                self.b_local_trace[it] = np.copy(self.b_local_trace[it-1])
#                self.LL_local_trace[it] = np.copy(self.LL_local_trace[it-1])
        else:
#            proposal_cov = self.scaling*(np.cov(np.vstack((self.a_local_trace[self.burnin:it],self.b_local_trace[self.burnin:it]))) + (1e-6)*np.eye(self.dim))
#            pdb.set_trace()
#            proposal_a,proposal_b = np.array((self.a_local_trace[it-1],self.b_local_trace[it-1])) + np.dot(np.linalg.cholesky(proposal_cov), np.random.normal(size=2))
#            proposal_a,proposal_b = proposal_theta
            proposal_a = np.random.normal(loc = self.a_local_trace[it-1],
                                          scale = self.proposal_tune)
            proposal_b = np.random.normal(loc = self.b_local_trace[it-1],
                                          scale = self.proposal_tune)
            proposal_LL = self.local_LL(proposal_a, proposal_b)
            accept_prob = proposal_LL - self.LL_local_trace[it-1]
            accept_prob += stats.multivariate_normal.logpdf(proposal_a, self.prior_mean, self.prior_var*np.eye(2)) - stats.multivariate_normal.logpdf(self.a_local_trace[it-1], self.prior_mean, self.prior_var*np.eye(2))
            accept_prob += stats.multivariate_normal.logpdf(proposal_b, self.prior_mean, self.prior_var*np.eye(2)) - stats.multivariate_normal.logpdf(self.b_local_trace[it-1], self.prior_mean, self.prior_var*np.eye(2))
            u = np.log(np.random.uniform())
            self.MH_prob_local[1] += 1.
            self.a_local_all_trace[it] = proposal_a
            self.b_local_all_trace[it] = proposal_b
            if u < accept_prob:
                self.a_local_trace[it] = proposal_a
                self.b_local_trace[it] = proposal_b
                self.LL_local_trace[it] = proposal_LL
                self.MH_prob_local[0] +=1.
            else:
                self.a_local_trace[it] = np.copy(self.a_local_trace[it-1])
                self.b_local_trace[it] = np.copy(self.b_local_trace[it-1])
                self.LL_local_trace[it] = np.copy(self.LL_local_trace[it-1])


    def local_ess(self,it):
        nu_a,nu_b = np.random.normal(scale = np.sqrt(self.prior_var), size=2)
        u = np.log(np.random.uniform())


        if it == 0:
            self.a_local_trace[it],self.b_local_trace[it] = np.random.normal(loc = self.prior_mean, scale = np.sqrt(self.prior_var), size=2)
            log_LL = self.local_LL(self.a_local_trace[it],self.b_local_trace[it]) + u
        else:
            log_LL = np.copy(self.LL_local_trace[it-1]) + u
            
        theta = np.random.uniform(0., 2.*np.pi)        
        bracket = [theta-(2.*np.pi),theta]
        
        if it == 0:
            proposal_a = np.copy(nu_a)*np.sin(theta) + self.a_local_trace[it]*np.cos(theta) + self.prior_mean[0]
            proposal_b = np.copy(nu_b)*np.sin(theta) + self.b_local_trace[it]*np.cos(theta) + self.prior_mean[1]
        else:
            proposal_a = np.copy(nu_a)*np.sin(theta) + self.a_local_trace[it-1]*np.cos(theta) + self.prior_mean[0]
            proposal_b = np.copy(nu_b)*np.sin(theta) + self.b_local_trace[it-1]*np.cos(theta) + self.prior_mean[1]


        self.LL_local_trace[it] = self.local_LL(proposal_a, proposal_b)
        while self.LL_local_trace[it] < log_LL:
            if theta < 0:
                bracket[0] = theta
            else:
                bracket[1] = theta
            theta = np.random.uniform(bracket[0],bracket[1])
            if it == 0:
                proposal_a = np.copy(nu_a)*np.sin(theta) + self.a_local_trace[it]*np.cos(theta) + self.prior_mean[0]
                proposal_b = np.copy(nu_b)*np.sin(theta) + self.b_local_trace[it]*np.cos(theta) + self.prior_mean[1]
            else:
                proposal_a = np.copy(nu_a)*np.sin(theta) + self.a_local_trace[it-1]*np.cos(theta) + self.prior_mean[0]
                proposal_b = np.copy(nu_b)*np.sin(theta) + self.b_local_trace[it-1]*np.cos(theta) + self.prior_mean[1]
    
            self.LL_local_trace[it] = self.local_LL(proposal_a, proposal_b)

        self.a_local_trace[it] = proposal_a
        self.b_local_trace[it] = proposal_b
        self.a_local_all_trace[it] = proposal_a
        self.b_local_all_trace[it] = proposal_b

                


    def full_LL(self,a,b):
        LL = (self.Z - self.Y)*(a + b*self.X)
        LL -= self.Z*np.log(1. + np.exp(a + b*self.X))
        LL = LL.sum()
        LL += (-1./(2.*self.prior_var)) * ((a-self.prior_mean[0])**2 + (b-self.prior_mean[1])**2)
        return(LL)                
        
    def local_LL(self,a,b):
        LL = (self.Z_local - self.Y_local)*(a + b*self.X_local)
        LL -= self.Z_local*np.log(1. + np.exp(a + b*self.X_local))
        LL = LL.sum()
#        LL += (-1./(2.*self.prior_var)) * ((a-self.prior_mean[0])**2 + (b-self.prior_mean[1])**2)
        return(LL)
        
    def final_MH(self,it):
        if it == 0:
            self.LL_hat_trace[it] = self.predict_LL(self.theta_mean_burnin)
            self.a_hat_trace[it] = self.theta_mean_burnin[0]
            self.b_hat_trace[it] = self.theta_mean_burnin[1]
        else:
            if self.random_walk:
                theta_star = np.array([self.a_hat_trace[it-1], self.b_hat_trace[it-1]]) +np.dot(np.linalg.cholesky(self.theta_cov_burnin), np.random.normal(size=(2,1)))
                proposal_a, proposal_b= np.copy(theta_star)
                theta_star = theta_star.reshape(1,-1)
                proposal_LL = self.predict_LL(theta_star) 
#                accept_prob = proposal_LL - self.LL_hat_trace[it-1]
                accept_prob = (proposal_LL+self.LL_approx_mean) - (self.LL_hat_trace[it-1]+self.LL_approx_mean) 
            else:
                proposal_a,proposal_b = self.dp.sample()[0][0]
                theta_star = np.array([proposal_a,proposal_b]).reshape(1,-1)            
                old_theta = np.array([self.a_hat_trace[it-1],
                                      self.b_hat_trace[it-1]]).reshape(1,-1)
                proposal_LL = self.predict_LL(theta_star)                 
#                accept_prob = (self.LL_approx_sd*proposal_LL+self.LL_approx_mean) - (self.LL_approx_sd*self.LL_hat_trace[it-1]+self.LL_approx_mean) 
                accept_prob = proposal_LL - self.LL_hat_trace[it-1]

                accept_prob += self.dp.score_samples(old_theta) - self.dp.score_samples(theta_star)
                accept_prob += stats.multivariate_normal.logpdf(theta_star, self.prior_mean, self.prior_var*np.eye(2)) - stats.multivariate_normal.logpdf(old_theta, self.prior_mean, self.prior_var*np.eye(2))
#                accept_prob += stats.multivariate_normal.logpdf(proposal_a, self.prior_mean, self.prior_var*np.eye(2)) - stats.multivariate_normal.logpdf(self.a_hat_trace[it-1], self.prior_mean, self.prior_var*np.eye(2))
#                accept_prob += stats.multivariate_normal.logpdf(proposal_b, self.prior_mean, self.prior_var*np.eye(2)) - stats.multivariate_normal.logpdf(self.b_hat_trace[it-1], self.prior_mean, self.prior_var*np.eye(2))

            u = np.log(np.random.uniform())
            self.MH_prob_final[1] +=1.
            if u < accept_prob:
                self.a_hat_trace[it] = proposal_a
                self.b_hat_trace[it] = proposal_b
                self.LL_hat_trace[it] = proposal_LL
                self.MH_prob_final[0] +=1.
            else:
                self.a_hat_trace[it] = self.a_hat_trace[it-1]
                self.b_hat_trace[it] = self.b_hat_trace[it-1]
                self.LL_hat_trace[it] = self.LL_hat_trace[it-1]
        
    def predict_LL(self, theta_star, h=.0005):
        theta_star_scale = self.scaler.transform(theta_star)
        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta_scale-theta_star_scale,(self.theta_scale-theta_star_scale)**2))
        W_i = (h/float(self.theta_n))*rbf_kernel(theta_star_scale.reshape(1,-1), self.theta_scale, gamma=h) * np.eye(self.theta_n)
#        R_i = np.hstack((np.ones((self.theta_n,1)), self.theta-theta_star,(self.theta-theta_star)**2))
#        W_i = h*rbf_kernel(theta_star.reshape(1,-1), self.theta, gamma=h) * np.eye(self.theta_n)
        RW = np.dot(R_i.T, W_i)
        RWR_inv = np.linalg.inv( np.dot(RW,R_i))
        RWLL = np.dot(RW, self.LL_approx)
        beta_hat = np.dot(RWR_inv, RWLL)
        return(beta_hat[0])

    def local_reg_CV(self,h=.0005):
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
        sigma_hat = np.mean((self.LL_approx[:-1] - self.LL_approx[1:])**2)
        sigma_hat /= 2.
        risk -= sigma_hat
        risk += 2.*sigma_hat*h/float(self.theta_n)
        return(risk)

if __name__ == '__main__':
    synthetic_data = loadmat("../data/synthetic_data.mat")
    pmc = parallelMCMC(X=synthetic_data['X'],Y=synthetic_data['Y'],
                       Z=synthetic_data['Z'],a=synthetic_data['a'][0][0],
                       b=synthetic_data['b'][0][0])
    pmc.sample()