NutrientMixingAlgorithm
=======================

A Python-based Algorithm for Nutrient Mixing in Hydro- and Aquaponics systmes created by the 
[Automatic Control & System Dynamics Lab](https://www.tu-chemnitz.de/etit/control/index.php.en "ACSD Lab") 
at Chemnitz University of Technology.

Introduction
------------

The program features an algorithm for automatically mixing a nutrient solution based on a target volume and reference concentrations for the nutrients that are considered by the user. To achieve the desired solution the algorithm optimally chooses the volumes of a set of water sources and masses of a set of fertilizers, that are flowing into a mixing tank. Both sets can be specified by the user of the program. 

The optimal choice of inputs (water volumes and fertilizer masses) is based on the solution of a convex static optimization problem

$$\begin{align}
\min_{u \in R^n}\ & f(u), \\
\text{s.t.}\ & g_L \le g(u) \le g_U, \\
& u_L \le u \le u_U.
\end{align}$$

The optimization problem features two objectives:

1. Minimized error between a given reference concentration and the resulting nutrient concentration in the mixed solution,
2. Optimized use of the given water sources and fertilizers -> specified by the user.

The priority of both objectives and the optimal ressource consumption can be affected by setting the weights of the optimization problem.

Implementation
--------------

The program is implemented as a python script (Version 3.11). The interface between the program and the user is done via xml-files. These files are set up as tables, in which every parameter of the optimization problem is stated. Parameters include:

- Weights to affect the optimal consumption of ressources and the priority of the objectives
- Reference concentrations, max/min concentrations of the final solution,
- Nutrient composition of every water source and fertilzer,
- Mixing tank geometry, ...

Reading from and writing to the xml-files is done via the pandas-library and the lxml-parser. With the help of the read in parameters the optimization problem is set up with the Python API of CasADi and Solved by the IPOPT-Solver. References can be found in the associated article that is presented at the end of the README-file.

A more extensive investigation into the structure of the program can be found in the documentation.

Citation
--------

We provide this program open source and hope it will help you for your research and/or for setting up your aquaponics, hydroponics or a similar system. However, if you do so, **please cite the following article in your work**:

* Kobelski, A.; Nestler, P.; Mauerer, M.; Rocksch, T.; Schmidt, U.; Streif, S. **[An Algorithm for Nutrient Mixing Optimization in Aquaponics.](https://doi.org/10.3390/app14188140)** Appl. Sci. 2024, 14, 8140.  



Note: The ReadMe-File is currently being edited.
