# Author: Batiste Le Bars
# Purpose: Implementation of the synthetic experiments from the paper "Refined Convergence and 
# Topology Learning for Decentralized SGD with Heterogeneous Data"

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

from STL_FW import *    # Import STL-FW


''' Directly run this script to get the results in two Json '''


##########################################################
###################### FUNCTIONS #########################
##########################################################




def obj_mean(PI,means,theta):

    ''' Objective function of the mean estimation problem '''

    return theta**2 - 2*theta*np.sum(means*np.mean(PI,0))




def DSGD(PI, sigma_2, means, W, theta0, eta, iter_max, mu_star):


    '''
    
     Decentralized SGD for the specific problem of mean estimation.
        
    '''


    f_obj = [[obj_mean(PI,means,theta0[j]) for j in range(n)]]

    error = [(1/n)*np.linalg.norm(theta0-mu_star,2)**2]

    error_min = [np.min(np.square(theta0-mu_star))]

    error_max = [np.max(np.square(theta0-mu_star))]


    for it in range(iter_max):

        for i in range(n):

            Y_i = np.random.multinomial(1,PI[i,:],1)
            mu_i = Y_i.dot(means)
            X_i = np.random.normal(mu_i,sigma_2)

            theta0[i] = theta0[i] - 2*eta*(theta0[i] - X_i) # gradient descent step

        
        theta0 = W.dot(theta0) # averaging
        
        f_obj += [[obj_mean(PI,means,theta0[j]) for j in range(n)]]

        error += [(1/n)*np.linalg.norm(theta0-mu_star,2)**2]

        error_min += [np.min(np.square(theta0-mu_star))]

        error_max += [np.max(np.square(theta0-mu_star))]

    return theta0, f_obj, error, error_min, error_max






def dir_param(k,K,p):

    alpha_dirichlet = 100*np.ones(K)
    alpha_dirichlet[k] = 100*p

    return alpha_dirichlet




def generation(K,m_diff):


    '''
    
    Give the true means and the matrix PI containing class proportions
    
    '''

    means = np.linspace(-m_diff,m_diff,K)

    PI = [np.random.dirichlet(dir_param(k,K,1000), int(n/K)) for k in range(K)]

    PI = np.concatenate(PI)

    return means, PI



def random_d_regular(n,budg):


    '''
    
    Construct a random d-regular graph of size n and d = budg
    
    '''


    A_rand = nx.adjacency_matrix(nx.random_regular_graph(budg,n)).todense(
                                                        ) + np.identity(n)
    W_rand = np.array((1/(budg+1))*A_rand)

    rho = eigh(W_rand.T.dot(W_rand), 
                    eigvals_only = True, subset_by_index = [n-2,n-2])[0] 

    while rho>=1:
        A_rand = nx.adjacency_matrix(
            nx.random_regular_graph(budg,n)).todense() + np.identity(n)
        W_rand = np.array((1/(budg+1))*A_rand)
        rho = eigh(W_rand.T.dot(W_rand),
                 eigvals_only = True, subset_by_index = [n-2,n-2])[0] 

    return W_rand




##########################################################
######################### LOOP ###########################
##########################################################




def loop(n, K, sigma_2_tilde, iter_max, n_xp, BUDG, HETERO, ETA):

    try:

        with open('./simulearnS.json') as json_file:
            Res = json.load(json_file)

        Res = json.loads(Res)

        with open('./simurandS.json') as json_file:
            Res_rand = json.load(json_file)

        Res_rand = json.loads(Res_rand)

    except FileNotFoundError:

        Res = {}
        Res_rand = {}        


    for budg in BUDG:

        for m_diff in HETERO:

            for eta in ETA:

                if str(budg)+'-'+str(m_diff)+'-'+str(eta) not in Res.keys():

                    Res[str(budg)+'-'+str(m_diff)+'-'+str(eta)] = {}
                    Res_rand[str(budg)+'-'+str(m_diff)+'-'+str(eta)] = {}

                    Error = []
                    Error_min = []
                    Error_max = []
                    Error_rand = []
                    Error_min_rand = []
                    Error_max_rand = []

                    for xp in range(n_xp):


                        means, PI = generation(K,m_diff)

                        mu_star = means.dot(np.mean(PI,0))

                        B = 4*np.square(m_diff-np.mean(means)) # B du papier


                        # Topologie learning 

                        sigma2 = 4*sigma_2_tilde


                        lambd = sigma2/(K*(B+(0.1)*(B==0)))

                        W, _ = frank_wolfe_budg(PI, lambd, budget = budg)

                        W_rand = random_d_regular(n,budg)

                    
                        # D-SGD pour FW


                        theta_init = 10*np.ones(n)


                        _ , f_obj, error, error_min, error_max = DSGD(PI, 
                                    sigma_2_tilde, means, W, theta_init, eta, iter_max, mu_star)

                        Error += [error]
                        Error_min += [error_min]
                        Error_max += [error_max]


                        # D-SGD pour Random

                        theta_init = 10*np.ones(n)


                        _ , f_obj_rand, error_rdn, error_min_rnd, error_max_rnd = DSGD(PI,
                                sigma_2_tilde, means, W_rand, theta_init, eta, iter_max, mu_star)

                        Error_rand += [error_rdn]
                        Error_min_rand += [error_min_rnd]
                        Error_max_rand += [error_max_rnd]



                    
                    Res[str(budg)+'-'+str(m_diff)+'-'+str(eta)]['MEAN'] = Error
                    Res[str(budg)+'-'+str(m_diff)+'-'+str(eta)]['MIN'] = Error_min
                    Res[str(budg)+'-'+str(m_diff)+'-'+str(eta)]['MAX'] = Error_max

                    Res_rand[str(budg)+'-'+str(m_diff)+'-'+str(eta)]['MEAN'] = Error_rand
                    Res_rand[str(budg)+'-'+str(m_diff)+'-'+str(eta)]['MIN'] = Error_min_rand
                    Res_rand[str(budg)+'-'+str(m_diff)+'-'+str(eta)]['MAX'] = Error_max_rand



                    json_string = json.dumps(Res)
                    with open('./simulearnS.json', 'w') as outfile:
                        json.dump(json_string, outfile)

                    json_string = json.dumps(Res_rand)
                    with open('./simurandS.json', 'w') as outfile:
                        json.dump(json_string, outfile)

            

                print(budg, m_diff, eta)












##########################################################
######################## PARAM ###########################
##########################################################


n = 100
K = 10


sigma_2_tilde = 1


iter_max = 150

n_xp = 10


BUDG = np.arange(2,11)  # communication budget, d_max in the paper

HETERO = [1, 5, 10, 15, 20, 25, 30, 35, 45, 50, 60, 75, 100] # heterogeneity lever, m in the paper

ETA = [0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.5, 1.0] # step size hyperparam, eta in the paper


#### Run big loop ####



loop(n, K, sigma_2_tilde, iter_max, n_xp, BUDG, HETERO, ETA)
