Code that implements the algorithm 'Sparse Topology Learning with Frank-Wolfe' (STL-FW) and the synthetic data experiments presented in the paper
'Refined Convergence and Topology Learning for Decentralized SGD with Heterogeneous Data'.


I. Contents

- STL_FW.py
- SimulationLoop.py
- PlotSimu.py

II. Requirements

- numpy
- scipy
- matplotlib
- json
- networkx
- mpl_toolkits

III. Code Information

- STL_FW.py contains the code for the algorithm 'Sparse Topology Learning with Frank-Wolfe'. It is used in the scripts SimulationLoop.py and PlotSimu.py.

- SimulationLoop.py can be run in order to obtain the 10 convergence results for all 'budget', 'degree of heterogeneity' and 'stepsize eta'.
  It creates two json files: simulearnS.json contains the results obtained with the topology learned by STL-FW and simurandS.json contains the results obtained with the random topology. 
  To reduce the calculation of SimulationLoop.py, the experiment parameters can be changed at the end of the script by reducing the list of 'budget',
  'degree of heterogeneity' and 'stepsize eta' to look at.

- PlotSimu.py can be run in order to get the plots of Figure 1 from the paper. Once run, it reads the json files and the console asks which plot
 (a, b or c) in the figure 1 to show.
