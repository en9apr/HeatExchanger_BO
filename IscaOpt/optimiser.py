#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Optimiser suite
================================================================================
:Author:
   Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
   19 April 2017
:Copyright:
   Copyright (c)  Alma Rahat, University of Exeter, 2017
:File:
   optimiser.py
"""

# imports
import numpy as np
import GPy as GP
import cma as CMA
try:
    import mono_surrogate
    import multi_surrogate
except:
    import IscaOpt.mono_surrogate as mono_surrogate
    import IscaOpt.multi_surrogate as multi_surrogate
import sys, time
import matplotlib.pyplot as plt


def mod_evaluate(x, toolbox):
    """
    Modify toolbox function to work with CMA_ES (Hansen). In CMA-ES the default 
    is to minimise a given function. Here we negate the given function to turn 
    it into a maximisation problem.
    
    Parameters.
    -----------
    x (np.array): decision vector
    toolbox (DEAP toolbox): toolbox with the infill cost function.
    """
    try:
        return -toolbox.evaluate(x)[0][0]
    except:
        return -toolbox.evaluate(x)[0]

class Optimiser(object):
    """
    A suite of optimisers where the inidividual optimisers may be called
    without creating an instance of the optimiser.
    """
        
    @staticmethod
    def CMA_ES(toolbox, centroid=[1], std_dev=1, cma_options={}):
        """
        A wrapper for CMA-ES (Hansen) with DEAP toolbox.
        
        Parameters.
        -----------
        toolbox (DEAP toolbox): an appropriate toolbox that consists of 
                                the objective function for CMA-ES.
        centroid (list or str): centroid vector or a string that can produce a 
                                centroid vector. 
        std_dev (float): standard deviation from the centroid.
        lb (list or np.array): lower bounds in the decision space. 
        ub (list or np.array): upper bounds in the decision space.
        cma_options (dict): cma_options from Hansen's CMA-ES. Consult the relevant 
                            documentation.
                            
        Returns the best approximation of the optimal decision vector.
        """
        func = mod_evaluate
        fargs = (toolbox,)
        res = CMA.fmin(func, centroid, std_dev, \
            options=cma_options, args=fargs, bipop=True, restarts=9)
        return res[0]
        
    @staticmethod
    def grid_search_1D(toolbox, lb, ub, n=10000, obj_sense=1):
        '''
        Grid search in one dimension. Evluate the objective function at equally
        spaced locations between the lower and upper boundary of the decision
        space.
        
        Parameters.
        -----------
        toolbox (DEAP toolbox): toolbox that contains the objective function.
        lb (np.array): lower bounds on the decision space.
        ub (np.array): upper bounds on the decision space.
        n (int): number of samples to evaluate.
        obj_sense (int): optimisation sense. Keys. 
                            -1: minimisation.
                             1: maximisation.
                             
        Returns the best decision vector from the observed solutions. 
        '''
        x = np.linspace(lb, ub, n)[:,None]
        y = np.array([toolbox.evaluate(i) for i in  x])
        opt_ind = np.argmax(obj_sense*y)
        return x[opt_ind]
    
    @staticmethod
    def EMO(func, fargs=(), fkwargs={}, cfunc=None, cargs=(), ckwargs={},\
            settings={}):
        '''
        Optimising a single- or multi-objective problem using 
        Gaussian process surrogate(s).
        
        Parameters. 
        -----------
        func (function): objective function.
        fargs (tuple): arguments to the objective function.
        fkwargs (dict): keyword arguments for the objective function.
        cfunc (function): cheap constraint function.
        cargs (tuple): arguments to the constraint function.
        ckwargs (dict): keyword arguments to the constraint fucntion.
        settings (dict): various settings for the Bayesian optimiser. 
        
        Returns a list of all the observed decision vectors, the associated 
        objective values, and relevant hypervolume.
        '''
        start_sim = time.time()
        # parameters for problem class
        n_dim = settings.get('n_dim', 2)
        n_obj = settings.get('n_obj', 2)
        lb = settings.get('lb', np.zeros(n_dim))
        ub = settings.get('ub', np.ones(n_dim))
        ref_vector = settings.get('ref_vector', [150.0]*n_obj)
        obj_sense = settings.get('obj_sense', [-1]*n_obj) # default: all minimisation; 
                                                          # haven't tried maximisation, but should work.
        method_name = settings.get('method_name', 'HypI')
        visualise = settings.get('visualise', False)
        # parameters for EGO
        n_samples = settings.get('n_samples', n_dim * 10) # default is 10 
                                                          # initial samples per dimension
        budget = settings.get('budget', n_samples+5)
        kern_name = settings.get('kern', 'Matern52')
        verbose = settings.get('verbose', True)
        svector = settings.get('svector', 15)
        maxfevals = settings.get('maxfevals', 5000*n_dim)
        multisurrogate = settings.get('multisurrogate', False)
        # history recording
        sim_dir = settings.get('sim_dir', '')
        run = settings.get('run', 0)
        draw_true_1d = settings.get('draw_true_1d', False)
        # cma_options for Hansen's CMA-ES
        cma_options = settings.get('cma_options', \
                                    {'bounds':[list(lb), list(ub)], \
                                     'tolfun':1e-7, \
                                     'maxfevals':maxfevals,\
                                     'verb_log': 0,\
                                     'CMA_stds': np.abs(ub - lb)}) 
        cma_centroid = '(np.random.random('+str(n_dim)+') * np.array('+\
                        str(list(ub - lb))+') )+ np.array('+str(list(lb))+')'
        cma_sigma = settings.get('cma_sigma', 0.25)
        # initial design file
        init_file = settings.get('init_file', None)
        X = None
        Y = None
        # intial training data
        if init_file is not None:
            data = np.load(init_file)
            X = data['arr_0']
            Y = data['arr_1']
            xtr = X.copy()
            print('Training data loaded from: ', init_file)
            print('Training data shape: ', xtr.shape)
        if verbose:
            print ("=======================")
        # initialise
        if verbose:
            print('Simulation settings. ')
            print(settings)
            print('=======================')
        hpv = []
        # determine method
        if multisurrogate:
            method = getattr(multi_surrogate, method_name)
            kern = [getattr(GP.kern, kern_name)(input_dim=n_dim, ARD=True) \
                        for i in range(n_obj)]
        else:
            method = getattr(mono_surrogate, method_name)
            kern = getattr(GP.kern, kern_name)(input_dim=n_dim, ARD=True)
        print(method, kern)
        # method specific kwargs
        skwargs = {}
        if method_name == 'HypI':
            skwargs['ref_vector'] = ref_vector
        if method_name == 'ParEGO':
            skwargs['s'] = svector
        if method_name == 'MPoI' or method_name == 'SMSEGO':
            skwargs['budget'] = budget
        print('method used: ', method.__name__)
        # EGO
        i = 1
        count_limit = (budget-n_samples)
        sim_file = sim_dir + func.__name__ + '_' + method_name + \
                                        '_b' + str(budget) + \
                                        's' + str(n_samples) \
                                        + '_r' + str(run) + '.npz'
        while True:
            print('Episode: ', i)
            mop = method(func,n_dim, n_obj, lb, ub, obj_sense=obj_sense, \
                        args=fargs, X=X, Y=Y, kwargs=fkwargs, kern=kern,\
                        ref_vector=ref_vector)
            print(mop.lower_bounds, mop.upper_bounds)
            if i == 1 and init_file is None:
                xtr = mop.lhs_samples(n_samples, cfunc, cargs, ckwargs)
                if xtr.shape[0] < n_samples:
                    count_limit += n_samples - xtr.shape[0] - 1
                    print("adjusted count limit: ", count_limit)
                    print("initial samples: ", xtr.shape[0])
            toolbox = mop.get_toolbox(xtr, skwargs, cfunc=cfunc, cargs=cargs, \
                                        ckwargs=ckwargs)
            hpv.append(mop.current_hv)
            X = mop.X.copy()
            Y = mop.Y.copy()
            
            if i > count_limit:
                break
            i += 1
            
            start = time.time()
            if n_dim > 1:
                xopt = Optimiser.CMA_ES(toolbox, cma_centroid, cma_sigma, \
                                    cma_options=cma_options)
            else:
                xopt = Optimiser.grid_search_1D(toolbox, lb, ub, n=maxfevals)
            
            print('Time taken (minutes): ', (time.time() -start)/60.0) 
            
            print('Best individual:')
            print(xopt)
            print('Best individual fitness:')
            print(toolbox.evaluate(xopt))
            
            # new sample point location from CMA-ES optimisation
            x_new = np.reshape(xopt, (1, -1))
            # debug stuff
            print("Next sampling points: ", x_new)
            
            xtr = mop.X.copy()
            # include new sample in the training data.
            xtr = np.concatenate([xtr, x_new])
            if visualise:
                if n_obj == 2:
                    plt.figure(1)
                    plt.cla()
                    y = Y.copy() #mop.m_obj_eval(xtr)
                    plt.scatter(y[:n_samples,0], y[:n_samples,1], marker="x", \
                                color="blue", alpha=0.35)
                    plt.scatter(y[n_samples:,0], y[n_samples:,1], \
                        c=np.arange(1, y.shape[0]-n_samples+1, 1), alpha=0.35)
                    plt.scatter(y[-1,0], y[-1,1], facecolor="none", \
                                edgecolor="black", s=80)
                    plt.xlabel("$f_1$")
                    plt.ylabel("$f_2$")
                    plt.draw()
                    plt.pause(0.005)
                elif n_obj == 1 and n_dim == 2:
                    plt.figure(1)
                    plt.cla()
                    y = Y.copy() #mop.m_obj_eval(xtr)
                    xtmp = X.copy()
                    plt.scatter(xtmp[:n_samples,0], xtmp[:n_samples,1], \
                                marker="x", color="blue", alpha=0.35)
                    plt.scatter(xtmp[n_samples:,0], xtmp[n_samples:,1], \
                        c=np.arange(1, xtmp.shape[0]-n_samples+1, 1), alpha=0.35)
                    plt.scatter(xtmp[-1,0], xtmp[-1,1], facecolor="none", \
                                edgecolor="black", s=80)
                    plt.xlabel('$x_0$')
                    plt.ylabel('$x_1$')
                    plt.draw()
                    plt.pause(0.005)
                elif n_obj == 1 and n_dim == 1:
                    plt.figure(1, figsize=(6,8))
                    plt.subplot(211)
                    plt.cla()
                    y = Y.copy() #mop.m_obj_eval(xtr)
                    xtmp = X.copy()
                    tx = np.linspace(lb, ub, maxfevals)[:,None]
                    if draw_true_1d:
                        ty = [func(sol)[0] for sol in tx]
                    pred_y, pred_s = mop.surr.predict(tx)
                    ei = mop.surr.expected_improvement(tx, obj_sense=-1, lb=lb, \
                            ub=ub, cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
                    plt.scatter(xtmp[:n_samples], y[:n_samples], marker="x", \
                                                    color="blue", alpha=0.75)
                    plt.scatter(xtmp[n_samples:], y[n_samples:], \
                        c=np.arange(1, xtmp.shape[0]-n_samples+1, 1), alpha=0.75)
                    plt.scatter(xtmp[-1], y[-1], \
                        facecolor="none", edgecolor="black", s=80)
                    if draw_true_1d:
                        plt.plot(tx, ty, ls='dashed', color="black", alpha=0.5)
                    plt.plot(tx, pred_y, color="red")
                    plt.fill_between(np.squeeze(tx), np.squeeze(pred_y-pred_s), \
                            np.squeeze(pred_y+pred_s), color="red", alpha=0.3)
                    plt.xlabel('x')
                    plt.ylabel('f(x)')
                    plt.draw()
                    plt.pause(0.005)
                    plt.subplot(212)
                    plt.cla()
                    plt.plot(tx, ei)
                    plt.xlabel('x')
                    plt.ylabel('E[I(x)]')
                    plt.draw()
                    plt.pause(0.005)
                else:
                    print('Visualisation is not available for the number of objectives or the number of input dimensions.')
                
            
            if verbose:
                print ("=======================")
            if n_obj > 1:
                print ('Hypervolume: ', hpv[-1])   
            else:
                print ('Best function value: ', hpv[-1])
            if i%1 == 0:
                print('Saving data...')
                try:
                    np.savez(sim_file, X, Y, hpv, (time.time()-start_sim)/60.0)
                    print('Data saved in file: ', sim_file)
                except Exception as e:
                    print(e)
                    print('Data saving failed.')
            
                  
        print('Saving data...')
        try:
            np.savez(sim_file, X, Y, hpv, (time.time()-start_sim)/60.0)
            print('Data saved in file: ', sim_file)
        except Exception as e:
            print(e)
            print('Data saving failed.')
        return X, Y, hpv
        
