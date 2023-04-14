# Author: Batiste Le Bars
# Purpose: Code for the plots of the synthetic experiments in the paper "Refined Convergence and 
# Topology Learning for Decentralized Optimization with Heterogeneous Data"

import numpy as np
import matplotlib.pyplot as plt

from STL_FW import *

import json
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt



with open('./simulearnS.json') as json_file:
    learn = json.load(json_file)

learn = json.loads(learn)

with open('./simurandS.json') as json_file:
    rand = json.load(json_file)

rand = json.loads(rand)







BUDG = np.arange(2,11)

HETERO = [1, 5, 10, 15, 20, 25, 30, 35, 45, 50, 60, 75, 100]

ETA = [0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.5, 1.0]

iter_max = 150






# BUDG = [3]  # communication budget, d_max in the paper

# HETERO = [1, 5, 10, 15, 20, 25, 30, 35, 45, 50, 60, 75, 100] # heterogeneity lever, m in the paper

# ETA = [0.03] 


# iter_max = 150



##########################################################
###################### FUNCTIONS #########################
##########################################################



def dir_param(k,K,p):

    alpha_dirichlet = 100*np.ones(K)
    alpha_dirichlet[k] = 100*p

    return alpha_dirichlet



def generation(K,m_diff):

    means = np.linspace(-m_diff,m_diff,K)

    PI = [np.random.dirichlet(dir_param(k,K,1000), int(n/K)) for k in range(K)]

    PI = np.concatenate(PI)

    return means, PI





def abscisse_hetero(budg, it_eta, it_eta2, save = False, size = (6.4,4.8)):


    mean_FW = []
    min_FW = []
    max_FW = []
    mean_rnd = []
    min_rnd = []
    max_rnd = []


    for m_diff in HETERO:


        find_eta = [np.mean(np.array(learn[str(budg)+'-'
            +str(m_diff)+'-'+str(eta)]['MEAN']),0)[it_eta] for eta in ETA]

        eta_best = ETA[np.argmin(find_eta)]

        mean_FW += [np.mean(np.array(learn[str(budg)+'-'+str(m_diff)+'-'+str(eta_best)]['MEAN']
                            ),0)[it_eta]]

        min_FW += [np.mean(np.array(learn[str(budg)+'-'+str(m_diff)+'-'+str(eta_best)]['MIN']
                            ),0)[it_eta]]

        max_FW += [np.mean(np.array(learn[str(budg)+'-'+str(m_diff)+'-'+str(eta_best)]['MAX']
                            ),0)[it_eta]]



        find_eta_rnd = [np.mean(np.array(rand[str(budg)+'-'
            +str(m_diff)+'-'+str(eta)]['MEAN']),0)[it_eta2] for eta in ETA]

        eta_best_rnd = ETA[np.argmin(find_eta_rnd)]



        mean_rnd += [np.mean(np.array(rand[str(budg)+'-'+str(m_diff)+'-'+str(eta_best_rnd)]['MEAN']
                            ),0)[it_eta]]

        min_rnd += [np.mean(np.array(rand[str(budg)+'-'+str(m_diff)+'-'+str(eta_best_rnd)]['MIN']
                            ),0)[it_eta]]

        max_rnd += [np.mean(np.array(rand[str(budg)+'-'+str(m_diff)+'-'+str(eta_best_rnd)]['MAX']
                            ),0)[it_eta]]



    plt.plot(HETERO, mean_FW , color = 'orangered', linewidth=2.2, label = 'STL-FW (ours)')
    plt.plot(HETERO, max_FW, linestyle = '--', color = 'orangered', linewidth=2)
    plt.plot(HETERO, min_FW, linestyle = '--', color = 'orangered', linewidth=2)

    plt.plot(HETERO, mean_rnd , color = 'royalblue', linewidth=2.2, label = 'Random')
    plt.plot(HETERO, max_rnd, linestyle = '--', color = 'royalblue', linewidth=2)
    plt.plot(HETERO, min_rnd, linestyle = '--', color = 'royalblue', linewidth=2)

    plt.legend(fontsize=18)
    #plt.title('Budget = '+str(budg))
    plt.xlabel('Heterogeneity parameter $m$', fontsize=17)
    plt.ylabel('Error after '+str(it_eta)+' iterations', fontsize=18)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.rcParams["figure.figsize"] = size


    if save:
        plt.savefig('./figures/Simu/error_after_'+str(it_eta)+'_budg'+str(budg)+'.pdf')

    plt.show()







def plot_obj_quantity_oneplot(K, m_diff, sigma_2_tilde, lambd = 'default', save = False, size = (6.4,4.8)):   


    means, PI = generation(K,m_diff)

    B = 4*np.square(m_diff-np.mean(means)) # B du papier


    # Topologie learning 

    sigma2 = 4*sigma_2_tilde


    if lambd == 'default':
        lambd = sigma2/(K*(B+(0.1)*(B==0)))


    W_iter, f_obj_learn = frank_wolfe_iter(PI, lambd, n_iter = 99)

    Pi_het = np.linalg.norm( np.sum([np.outer((PI[:,k] - np.mean(PI[:,k])),PI[:,k]
                            ) for k in range(K)],0), 'nuc')   # norm nucléaire du thme 2


    Biais = [np.mean(4*np.square(np.sum((means - np.mean(means))*(W_iter[ii,:,:].dot(PI)
                            - np.mean(PI,0)),1))) for ii in range(W_iter.shape[0])]

    Biais = [np.mean(4*np.square(np.sum((means - np.mean(means))*(np.identity(n).dot(PI)
                            - np.mean(PI,0)),1)))] + Biais

    Variance = [np.linalg.norm(W_iter[ii,:,:] - (1/n)*np.ones((n,n)),
                                    'fro')**2 for ii in range(W_iter.shape[0])]

    Variance = [np.linalg.norm(np.identity(n) - (1/n)*np.ones((n,n)),
                                    'fro')**2] + Variance

    RHO = [eigh(np.identity(n).T.dot(np.identity(n)), 
                        eigvals_only = True, subset_by_index = [n-2,n-2])[0] ]

    RHO += [eigh(W_iter[ii,:,:].T.dot(W_iter[ii,:,:]), 
                        eigvals_only = True, subset_by_index = [n-2,n-2])[0] for ii in range(W_iter.shape[0])]

    Tau2 = np.array(Biais) + (sigma2/n)*np.array(Variance)





    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
                                            offset=(offset, 0))

    par2.axis["right"].toggle(all=True)

    host.set_xlim(-2, 101)
    host.set_ylim(-0.05, 1.45)

    host.set_xlabel("Iteration $l$", fontsize=18)
    host.set_ylabel("Topology Learning Objective", fontsize=18)
    par1.set_ylabel("Bias term of $H$", fontsize=18)
    par2.set_ylabel("Mixing parameter $1-p$", fontsize=18)

    p1, = host.plot(np.arange(len(RHO)), f_obj_learn, label="$g(W^{(l)})$", color = 'crimson')
    p2, = par1.plot(np.arange(len(RHO)), Biais, label="Bias term", color = 'forestgreen')
    p3, = par2.plot(np.arange(len(RHO)), RHO, label="1-p", color = 'darkorange')

    par1.set_ylim(-0.5, 41)
    par2.set_ylim(-0.05, 1.05)

    host.legend(fontsize=18)

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())

    host.tick_params(axis = 'y', colors = p1.get_color(), labelcolor = p1.get_color(), labelsize = 18)
    host.spines['left'].set_color(p1.get_color()) 

    par1.tick_params(axis = 'y', colors = p2.get_color())
    par1.spines['right'].set_color(p2.get_color()) 

    par2.tick_params(axis = 'y', colors = p3.get_color())
    par2.spines['right'].set_color(p3.get_color()) 

    par1.axis["right"].toggle(all=True)

    plt.grid(linestyle='--', linewidth=0.5)

    plt.rcParams["figure.figsize"] = size

    plt.draw()

    if save:
        plt.savefig('./figures/Simu/allquantities.pdf')

    plt.show()






############################
## WHICH FIGURE TO PLOT ? ##
############################



ffig = input("Enter which plot of figure 1 to show (a, b or c): \n ")


if ffig == 'a' :

    n = 100
    K = 10



    lambd = 0.5
    m_diff = 5
    sigma_2_tilde = 1


    plot_obj_quantity_oneplot(K, m_diff, sigma_2_tilde, lambd, save = False, size = (7,3.8))

elif ffig == 'b' :


    it_eta = 50

    budg = 3


    abscisse_hetero(budg, it_eta, it_eta, save = False, size = (8.5,5.5))


elif ffig == 'c' :


    it_eta = 50

    budg = 9


    abscisse_hetero(budg, it_eta, it_eta, save = False, size = (8.5,5.5))






