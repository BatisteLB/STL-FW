# Author: Batiste Le Bars
# Purpose: Implementation of STL-FW from the paper "Refined Convergence and 
# Topology Learning for Decentralized SGD with Heterogeneous Data"

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh



def obj_function(W,PI,lamb):

    '''
    
    Compute the value of the objective function g(W) (Eq 8 of the paper)

    '''

    nn = PI.shape[0]
    val = (1/nn)*np.linalg.norm(W.dot(PI)-(1/nn)*np.ones((nn,nn)).dot(PI),'fro')**2
    val = val + (lamb/nn)*np.linalg.norm(W-(1/nn)*np.ones((nn,nn)),'fro')**2
    return val



def frank_wolfe_iter(PI, lamb, n_iter = 'default'):

    '''
    
     STL-FW v1: Outputs all the iterates from 0 to n_iter.
     Useful if you don't want to re-run the algorithm for many different budget value but
    it has 'important' memory costs.
    

    Input:

    PI: A (n x K) array containing the class proportions of each node

    lamb: The (positive) hyperparameter quantifying the ratio variance/heterogeneity

    n_iter: Number of iterations. Default = 10*K



    Outputs:

    W_iter: A (n_iter x n x n) array containing the graph learned at each iteration

    function_value: A (n_iter) array containing the value of the objective at each iteration


    '''

    n = PI.shape[0]
    K = PI.shape[1]

    if n_iter == 'default':
        n_iter = 10*K

    function_value = np.zeros(n_iter + 1)

    W_FW = np.identity(n)
    W_iter = np.zeros((n_iter,n,n))


    for it in np.arange(n_iter):

        #print(it)

        function_value[it] = obj_function(W_FW,PI,lamb)

        # Computation of the gradient

        vv = W_FW.dot(PI)-np.repeat(np.mean(PI,0).reshape((1,-1)),n,0)

        vvv = np.sum([np.outer(vv[:,k],PI[:,k]) for k in range(K)],0)

        grad = (2/n)*vvv + (2/n)*lamb*(W_FW - (1/n)*np.ones((n,n)))

        # Find the best permutation matrix

        ind = linear_sum_assignment(grad)

        P = np.zeros((n,n))
        P[ind] = 1

        # Calculate line-search

        uu = np.mean(PI,0)-W_FW.dot(PI)
        uuu = (P-W_FW).dot(PI)

        num = np.sum([np.dot(uu[:,k],uuu[:,k]) for k in range(K)]) - lamb*np.trace(
                                        (W_FW - (1/n)*np.ones((n,n))).T.dot(P-W_FW))

        den = np.linalg.norm((P-W_FW).dot(PI),'fro')**2 + lamb*np.linalg.norm(P-W_FW,'fro')**2

        gamma = num/den

        if ((gamma < 0) or (gamma>1)):
            print('Check line-search')
        

        # Update 

        W_FW = (1-gamma)*W_FW + gamma*P

        W_iter[it,:,:] = W_FW    

    function_value[n_iter] = obj_function(W_FW,PI,lamb)

    return W_iter, function_value





def frank_wolfe_budg(PI, lamb, budget = 'default', n_max = 100):

    '''
    
     STL-FW v2: The algorithm runs until the maximal budget is attained. 
     Outputs only the final graph.
    

    Input:

    PI: A (n x K) array containing the class proportions of each node

    lamb: The (positive) hyperparameter quantifying the ratio variance/heterogeneity

    budget: Maximal number of edges per node to reach. Default = K.

    n_max: Maximal number of iteration (necessary if the budget is e.g. n-1). Default = 100.



    Outputs:

    W_FW: A (n x n) array containing the graph learned at the end

    function_value: An array containing the value of the objective at each iteration


    '''

    n = PI.shape[0]
    K = PI.shape[1]

    if budget == 'default':
        budget = K

    function_value = []

    W_FW_temp = np.identity(n)
    b = 0
    it = 0

    while ((b<=budget) and (it<=n_max)):

        W_FW = W_FW_temp

        #print(it)

        function_value += [obj_function(W_FW,PI,lamb)]

        # Computation of the gradient

        vv = W_FW.dot(PI)-np.repeat(np.mean(PI,0).reshape((1,-1)),n,0)

        vvv = np.sum([np.outer(vv[:,k],PI[:,k]) for k in range(K)],0)

        grad = (2/n)*vvv + (2/n)*lamb*(W_FW - (1/n)*np.ones((n,n)))

        # Find the best permutation matrix

        ind = linear_sum_assignment(grad)

        P = np.zeros((n,n))
        P[ind] = 1

        # Calculate line-search

        uu = np.mean(PI,0)-W_FW.dot(PI)
        uuu = (P-W_FW).dot(PI)

        num = np.sum([np.dot(uu[:,k],uuu[:,k]) for k in range(K)]) - lamb*np.trace(
                                            (W_FW - (1/n)*np.ones((n,n))).T.dot(P-W_FW))

        den = np.linalg.norm((P-W_FW).dot(PI),'fro')**2 + lamb*np.linalg.norm(P-W_FW,'fro')**2

        gamma = num/den

        if ((gamma < 0) or (gamma>1)):
            print('Check line-search')
        

        # Update 

        W_FW_temp = (1-gamma)*W_FW + gamma*P

        b = np.max(np.sum(W_FW_temp - np.identity(n) > 0,1))

        it += 1
   

    return W_FW, np.array(function_value)






def find_W(W_iter,budg):

    '''
    
    Function that takes the iterate of matrices learned with 'frank_wolfe_iter' and outputs
    the one satisfying the given budget.
    
    '''

    b = 0
    it = -1
    n = W_iter.shape[1]

    while ((b <= budg) and (it < W_iter.shape[0]-1)):

        it += 1

        tempW = W_iter[it,:,:]
    
        # Calculate budget

        b = np.max(np.sum(tempW - np.identity(n) > 0,1))


    # Calculate second largest eigenvalue

    tempW = W_iter[it-1,:,:]

    rho = eigh(tempW.T.dot(tempW), eigvals_only = True, subset_by_index = [n-2,n-2])[0]

    if rho == 1:
        print('Warning: Not enought connectivity. The algorithm may not converge')

    return tempW




