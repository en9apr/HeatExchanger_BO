"""
================================================================================
Exeter CFD Test Problems.

In this script, we show how to use the suite.
================================================================================
:Author:
    Steven Daniels <S.Daniels@exeter.ac.uk>
    Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
    26 August, 2018
:Copyright:
   Copyright (c)  Steven Daniels and Alma Rahat, University of Exeter, 2018
:File:
   sample_script.py
"""
import sys, time
import numpy as np
seed = 1435
np.random.seed(seed)

# APR added
import os
import subprocess
current = os.getcwd() 
from pyDOE import lhs as LHS  
import IscaOpt
# APR added

# import problem classes
import Exeter_CFD_Problems as TestProblems






## APR added sampling
#def lhs_initial_samples(n_dim, ub, lb, n_samples=4, cfunc=None, cargs=(), ckwargs={}):
#    """
#    Generate Latin hypercube samples from the decision space using pyDOE.
#
#    Parameters.
#    -----------
#    n_samples (int): the number of samples to take. 
#    cfunc (method): a cheap constraint function.
#    cargs (tuple): arguments for cheap constraint function.
#    ckawargs (dictionary): keyword arguments for cheap constraint function. 
#
#    Returns a set of decision vectors.         
#    """
#    seed = 1435
#    np.random.seed(seed)
#    samples = LHS(n_dim, samples=n_samples)
#    
#    scaled_samples = ((ub - lb) * samples) + lb            
#        
#    if cfunc is not None: # check for constraints
#        print('Checking for constraints.')
#        scaled_samples = np.array([i for i in scaled_samples if cfunc(i, *cargs, **ckwargs)])
#
#    return scaled_samples
## APR added sampling











if __name__=='__main__':
    if '-h' in sys.argv:
        print("This is a sample script that demonstrates how to run the test problems. The parameters may be configured with following arguments.")
        print("'-p': Mandatory argument that defines the problem name. The options are: 'PitzDaily', 'KaplanDuct' and 'HeatExchanger'")
        print("'-v': The CFD simulations under the bonnet produces a lot of text to show status of the simulation. We have tried to suppress many of these, but if you would like to see some of what is going on, please provide this argument.")
        sys.exit()
    if '-v' in sys.argv:
        verbose = True
    else:
        verbose = False
    if '-p' in sys.argv:
        problem_name = sys.argv[sys.argv.index('-p') + 1]
    else:
        raise ValueError('No valid problem name was defined. Please issue the following command to see help documentation: run sample_script.py -h')
    sys.argv = sys.argv[:1] # this is required for PyFoam to work correctly
    if problem_name == 'PitzDaily':
        print('Demonstration of the PitzDialy test problem.')
        # set up directories.
        settings = {
            'source_case': 'Exeter_CFD_Problems/data/PitzDaily/case_fine/',
            'case_path': 'Exeter_CFD_Problems/data/PitzDaily/case_single/',
            'boundary_files': ['Exeter_CFD_Problems/data/PitzDaily/boundary.csv'],
            'fixed_points_files': ['Exeter_CFD_Problems/data/PitzDaily/fixed.csv']
        }
        # instantiate the problem object
        prob = TestProblems.PitzDaily(settings)
        # get the lower and upper bounds
        lb, ub = prob.get_decision_boundary()
        # generate random solutions satisfying the lower and upper bounds.
        x = np.random.random((1000, lb.shape[0])) * (ub - lb) + lb
        rand_x = []
        for i in range(x.shape[0]):
            if prob.constraint(x[i]): # check to see if the random solution is valid
                rand_x.append(x[i])
        # evaluate for a solution
        print('Number of control points: ', prob.n_control)
        print('Decision vector: ', rand_x[0])
        print('Running simulation ...')
        start = time.time()
        res = prob.evaluate(rand_x[0], verbose=verbose)
        print('Objective function value:', res)
        print('Time taken:', time.time()-start, ' seconds.')
    elif problem_name == 'KaplanDuct':
        print('Demonstration of the KaplanDuct test problem.')
        # set up directories.
        settings = {
            'source_case': 'Exeter_CFD_Problems/data/KaplanDuct/case_fine/',
            'case_path': 'Exeter_CFD_Problems/data/KaplanDuct/case_single/',
            'boundary_files': ['Exeter_CFD_Problems/data/KaplanDuct/boundary_1stspline.csv', \
                                'Exeter_CFD_Problems/data/KaplanDuct/boundary_2ndspline.csv'],
            'fixed_points_files': ['Exeter_CFD_Problems/data/KaplanDuct/fixed_1.csv',\
                                'Exeter_CFD_Problems/data/KaplanDuct/fixed_2.csv']
        }
        # instantiate the problem object
        prob = TestProblems.KaplanDuct(settings)
        lb, ub = prob.get_decision_boundary()
        x = np.random.random((1000, lb.shape[0])) * (ub - lb) + lb
        rand_x = []
        for i in range(x.shape[0]):
            if prob.constraint(x[i]):
                rand_x.append(x[i])
        print('Number of control points: ', prob.n_control)
        print('Decision vector: ', rand_x[0])
        print('Running simulation ...')
        start = time.time()
        res = prob.evaluate(rand_x[0], verbose=verbose)
        print('Objective function value:', res)
        print('Time taken:', time.time()-start, ' seconds.')
    elif problem_name == 'HeatExchanger':
        print('Demonstration of the HeatExchanger test problem.')
        # set up directories.
        settings = {
            'source_case': 'Exeter_CFD_Problems/data/HeatExchanger/heat_exchange/',
            'case_path': 'Exeter_CFD_Problems/data/HeatExchanger/case_multi/'
        }
        # instantiate the problem object
        prob = TestProblems.HeatExchanger(settings)
        lb, ub = prob.get_decision_boundary()
        
        # reference vector
        ref = [0, 1e6]
        
        n_dim=lb.shape[0]
        n_samples = 11*n_dim-1
        #n_samples = 3
        print('Number of dimensions: ', n_dim)
        
        # the name of the sim_file is initial_samples.npz    
        sim_file = 'initial_samples.npz'

        # settings
        settings = {\
        'n_dim': n_dim,\
        'n_obj': 2,\
        'lb': lb,\
        'ub': ub,\
        'ref_vector': ref,\
        'method_name': 'HypI',\
        'budget':225,\
        'n_samples':220,\
        'visualise':False,\
        'multisurrogate':False, \
        'init_file':'initial_samples.npz'} # APR changed: n_dim, n_obj, lb, ub, deleted ref_vector, method to EGO from HypI, budget, n_samples
                
        res = IscaOpt.Optimiser.EMO(func=prob.evaluate, fargs=(), fkwargs={}, \
                            cfunc=prob.constraint, cargs=(), ckwargs={}, \
                            settings=settings)   
            
            
            
            
            