#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Base class for Multi Objective Bayesian optimisation
================================================================================
:Author:
   Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
   19 April 2017
:Copyright:
   Copyright (c)  Alma Rahat, University of Exeter, 2017
:File:
   BO_base.py
"""

# imports
import numpy as np
from pyDOE import lhs as LHS
from deap import base
from deap import creator
import _csupport as CS
from evoalgos.performance import FonsecaHyperVolume as FH

class BayesianOptBase(object):
    """
    Base class for mono- and multi-surrogate Bayesian optimiser. 
    """

    def __init__ (self, func, n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    ref_vector=None):
        """This constructor creates the base class.
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): tuple of arguments the objective function requires
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        ref_vector (np.array): reference vector in the objective space.
        """
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.obj_sense = obj_sense
        self.X = X
        self.Y = Y
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.ref_vector = ref_vector
    
    def lhs_samples(self, n_samples=4, cfunc=None, cargs=(), ckwargs={}):
        """Generate Latin hypercube samples from the decision space using pyDOE.

        Parameters.
        -----------
        n_samples (int): the number of samples to take. 
        cfunc (method): a cheap constraint function.
        cargs (tuple): arguments for cheap constraint function.
        ckawargs (dictionary): keyword arguments for cheap constraint function. 

        Returns a set of decision vectors.         
        """
        samples = LHS(self.n_dim, samples=n_samples)
        scaled_samples = ((self.upper_bounds - self.lower_bounds) * samples) \
                            + self.lower_bounds
        if cfunc is not None: # check for constraints
            print('Checking for constraints.')
            scaled_samples = np.array([i for i in scaled_samples \
                                            if cfunc(i, *cargs, **ckwargs)])
        return scaled_samples 
        
    def m_obj_eval(self, x):
        """Evaluate the expensive function. It ignores the decision vectors that
        have already been evaluated. 
        
        Parameters. 
        -----------
        x (np.array): set of decision vectors.
        
        Returns a set of ojective fucntion values.
        """
        if self.X is None: # the class was not initiated with a set of 
                           # initial decision vectors. 
            print('new X')
            self.X = x.copy() # local copy
            # evaluate given objective function
            y = np.array([self.func(i, *self.args, **self.kwargs) for i in x])
            if len(y.shape)<2:
                y = y[:,None]
            self.Y = y.copy()
            return self.Y
        else:
            shape = x.shape
            assert len(shape) == 2 # check for dimensionality requirements
            curr_shape = self.X.shape
            # check for solutions that have already been evaluated
            e_inds = [i for i in range(shape[0]) \
                if np.any(np.all(np.abs(x[i] - self.X) <= 1e-9, axis=1))]
            inds = [i for i in range(shape[0]) if i not in e_inds]
            if len(inds) > 0: # evaluate solutions that have not been evaluated
                y = np.array([self.func(i, *self.args, **self.kwargs) for i in x[inds]])
                if len(y.shape)<2:
                    y = y[:,None]
                self.Y = np.concatenate([self.Y, y], axis=0)
                self.X = np.concatenate([self.X, x[inds]], axis=0)
            else:
                print("No new evaluations were performed. This may happen when we start with a predefined set of solutions. If that is not the cause then this may be because the same solution has been selected. This is weired. If this happens, we need to see why this is happenning.")
                print(self.X.shape, x.shape, inds, self.Y.shape)
            return self.Y.copy()
            
    def get_dom_matrix(self, y, r=None):
        """
        Build a dominance comparison matrix between all observed solutions. Cell 
        keys for the resulting matrix.
        -1: The same solution, hence identical
         0: Row dominates column.
         1: Column dominates row. 
         2: row is equal to column.
         3: row and col mutually non-dominated.
         
        Parameters.     
        -----------
        y (np.array): set of objective vectors.
        r (np.array): reference vector
        """
        if r is not None:
            yr = np.append(y, [r], axis=0) #append the reference point at the end.
        else:
            yr = y
        n_data, n_obj = yr.shape
        redundancy = np.zeros((n_data, n_data))-1 # -1 means its the same solution.
        redundant = np.zeros(n_data)
        for i in range(n_data):
            for j in range(n_data):
                if i!= j:
                    redundancy[i, j] = CS.compare_solutions(yr[i], yr[j], \
                                                            self.obj_sense)
        return yr, redundancy
        
    def get_front(self, y, comp_mat=None, del_inds=None):
        """
        Get the Pareto front solution indices. 
        
        Parameters.
        -----------
        y (np.array): objective vectors.
        comp_mat (np.array): dominance comparison matrix.
        del_inds (np.array): the indices to ignore.
        """
        if comp_mat is None:
            yr, comp_mat = self.get_dom_matrix(y)
        else:
            yr = y
        dom_inds = np.unique(np.where(comp_mat == 1)[0])
        if del_inds is None:
            ndom_inds = np.delete(np.arange(comp_mat.shape[0]), dom_inds)
        else:
            ndom_inds = np.delete(np.arange(comp_mat.shape[0]), \
                            np.concatenate([dom_inds, del_inds], axis=0))
        return ndom_inds
        
    def init_deap(self, eval_function, obj_sense=1, lb=None, ub=None, \
                    cfunc=None, cargs=(), ckwargs={}):
        """Generate a DEAP toolbox so that optimisers may be used with the 
        problem. 

        Parameters. 
        -----------
        eval_function (method): infill criterion function 
        obj_sense (int): whether to maximise or minimise the infill criteria. 
                    keys.
                        1: maximisarion 
                       -1: minimisation (ignore this as this was not tested.)
        lb (np.array): lower bounds
        ub (np.array): upper bounds
        cfunc (method): cheap constraint function
        cargs (tuple): arguments for constraint function 
        ckwargs (dictionary): keyword argumnets for constraint function 

        Returns a toolbox in DEAP.        
        """
        # by default we maximise the infill criterion
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        if lb is not None: # this is for mono-surrogate
            toolbox.register("evaluate", eval_function, \
                        obj_sense=obj_sense, lb=lb, ub=ub, \
                        cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
        else: # this is for multi-surrogate
            toolbox.register("evaluate", eval_function, \
                        cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
        return toolbox    
    
    def current_hpv(self):
        """
        Calcualte the current hypervolume. 
        """
        y = self.Y
        if self.n_obj > 1:
            n_data = self.X.shape[0]
            y, comp_mat = self.get_dom_matrix(y, self.ref_vector)
            front_inds = self.get_front(y, comp_mat)
            hpv = FH(self.ref_vector)
            return hpv.assess_non_dom_front(y[front_inds])
        else:
            return np.min(y)
            
    def scalarise(self, args=()):
        """
        This is the infill criterion that should be implemented in the child class. 
        """
        raise NotImplementedError("Subclass must implement abstract method.")
        
    def get_toolbox(self, args=()):
        """
        This method should help a child class to generate appropriate toolbox 
        within DEAP. 
        """    
        raise NotImplementedError("Subclass must implement abstract method.")
