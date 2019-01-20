# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:43:12 2017

@author: KimuKook
"""
import numpy as np
import scipy
from functools import partial
from dogs import interpolation
from dogs import Utils
from dogs import constantK
from dogs import cartesian_grid
from dogs import Dogsplot

# This script shows the Delta DOGS(Lambda) - Constant K main code
n = 1              # Dimenstion
fun_arg = 2        # Type of function evaluation
# Initialize the grid size and parameters for Discrete search and Continuous search.
Nm = 8             # Initial mesh grid size
K = Kini = 3       # Continous search function constant
L0 = 1             # Discrete 1 search functio n constant
L = L0
# Initialize the index for iteration
nff = 1            # Number of experiments
num_iter = 0       # Represents how many iteration the algorithm goes
iter_max = 50      # Maximum number of iterations based on each mesh
MeshSize = 8       # Represents the number of mesh refinement that algorithm will perform
sc = 'constantK'   # represent the type of continuous search function we used.
alg_name = 'DDOGS/' # represents the name of optm algorithm

# plot class
# plot class
plot_index = 1
original_plot = 0
illustration_plot = 0
interpolation_plot = 0
subplot_plot = 0
store_plot = 1     # The indicator to store ploting results as png.

fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
func_eval = partial(Utils.fun_eval, fun, lb, ub)
    
Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

for ff in range(nff):
    newadd = []                  # Identify the point already exists in xU or not.
    L_refine = 0
    xU = Utils.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)

    if n == 1:
        xE = np.array([[0, 1]])
    elif n == 2:
        xE = np.array([[0.875, 0.5, 0.25, 0.125, 0.75, 0.5], [0.25, 0.125, 0.875, 0.5, 0.875, 0.75]])
    else:
        xE = np.random.rand(n, n + 1)
        xE = np.round(xE * Nm) / Nm  # quantize the points
    num_ini = xE.shape[1]
    yE = np.zeros(xE.shape[1])
    for ii in range(xE.shape[1]):
        yE[ii] = fun(xE[:, ii].reshape(-1, 1))

    if plot_index and n == 2:
        plot_parameters = {'original': original_plot, 'illustration': illustration_plot,
                           'subplot': subplot_plot, 'interpolation': interpolation_plot,
                           'store': store_plot}
        plot_class = Dogsplot.plot(plot_parameters, sc, num_ini, ff, fun_arg, alg_name)

    inter_par = interpolation.Inter_par(method="NPS")
    yE_best = np.array([])
    
    for kk in range(MeshSize):
        for k in range(iter_max):
            num_iter += 1
            print('Total iter = ', num_iter, 'iteration k = ', k, ' kk = ', kk, 'Nm = ', Nm)
            
            [inter_par, yp] = interpolation.interpolateparameterization(xE, yE, inter_par)
            K0 = np.ptp(yE, axis=0)
            # Calculate the discrete function.
            ypmin  = np.amin(yp)
            ind_min = np.argmin(yp)

            if ind_min != ind_min:
                print('~~~~~~~~~~~~~~~~~~~~')
            else:
                # Calcuate the discrete search function values at unevaluated support points:
                yu = np.zeros([1, xU.shape[1]])
                if xU.shape[1] != 0:
                    for ii in range(xU.shape[1]):
                        yu[0, ii] = interpolation.interpolate_val(xU[:, ii], inter_par) - L * Utils.mindis(xU[:, ii], xE)[0]

                if xU.shape[1] != 0 and np.amin(yu) < y0:  # or np.amin(yu) < 0
                    t = np.amin(yu)
                    ind = np.argmin(yu)
                    xc_eval = np.copy(xU[:, ind].reshape(-1, 1))
                    yc = -np.inf
                    xU = scipy.delete(xU, ind, 1)
                else:
                    while 1:
                        xs, ind_min = cartesian_grid.add_sup(xE, xU, ind_min)
                        xc, yc = constantK.tringulation_search_bound_constantK(inter_par, xs, K * K0, ind_min)
                        if interpolation.interpolate_val(xc, inter_par) < min(yp):
                            xc_grid = np.round(xc * Nm) / Nm
                            xc_eval = np.copy(xc_grid)
                            break
                        else:
                            xc_grid = np.round(xc * Nm) / Nm
                            if Utils.mindis(xc_grid, xE)[0] < 1e-6:
                                break
                            xc_grid, xE, xU, success, _ = cartesian_grid.points_neighbers_find(xc_grid, xE, xU, Bin, Ain)
                            if success == 1:
                                break
                            else:
                                yu = np.hstack((yu, (interpolation.interpolate_val(xc_grid, inter_par) - L * Utils.mindis(xc_grid, xE)[0]) ))
                    # The following statement represents inactivate step:
                    if Utils.mindis(xc_grid, xE)[0] < 1e-6:  # avoid true divide when calculate sd(xc).
                        L_refine = 1
                        # paper criteria (4)
                    else:
                        if newadd == 0:
                            xc_eval = np.copy(xc_grid)
                            newadd = []  # Point found from xU that already exists, delete that point in xU
                            dis, index, xu = Utils.mindis(xc_eval, xU)
                            xU = scipy.delete(xU, index, 1)
                            # The following statement represents activate step:
                        else:
                            # sd_xc = (interpolation.interpolate_val(xc, inter_par) - y0) / Utils.mindis(xc, xE)[0]
                            sd_xc = interpolation.interpolate_val(xc, inter_par) - L * Utils.mindis(xc_grid, xE)[0]
                            if xU.shape[1] != 0 and sd_xc > np.amin(yu):
                                # paper criteria (2)
                                ind = np.argmin(yu)
                                xc_eval = np.copy(xU[:, ind].reshape(-1, 1))
                                yc = -np.inf
                                xU = scipy.delete(xU, ind, 1)
                                print('Evaluate support point')
                            else:  # sd_xc < np.amin(yu) 
                                # paper criteria (3)
                                xc_eval = xc_grid

                # Minimize S_d ^ k(x)

                if L_refine == 1:
                    K *= 2
                    Nm *= 2
                    L += L0
                    L_refine = 0
                    print('===============  MESH Refinement  ===================')
                    print('===============  MESH Refinement  ===================')
                    break
                else:
                    xE = np.hstack([xE, xc_eval])
                    yE = np.hstack((yE, fun(xc_eval)))

            try:
                summary = {'alg': 'ori', 'xc_grid': xc_grid, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
                Dogsplot.print_summary(num_iter, k, kk, xE, yE, summary)
                if plot_index:
                    plot_class = Dogsplot.plot_delta_dogs(xE, Nm, plot_class, alg_name, Kini)
            except:
                print('Support points')
    Alg = {'name': alg_name}
    Dogsplot.save_data(xE, yE, inter_par, ff, Alg)
    Dogsplot.save_results_txt(xE, yE, fname, y0, xmin, Nm, Alg, ff, num_ini)
    Dogsplot.dogs_summary_plot(xE, yE, y0, ff, xmin, fname, alg_name)


##########################  Show the result  ############################
#data = np.hstack((xE.T, np.array([yE]).T))
#if n == 3:
#    data = np.vstack((np.array(['xE1', 'xE2', 'xE3', 'yE']), data))
#elif n == 2:
#    data = np.vstack((np.array(['xE1', 'xE2', 'yE']), data))
#else:
#    data = np.vstack((np.array(['xE', 'yE']), data))
#
#print(data)
#print(' Nm = ', Nm)
#print('minimum point = ', xE[:,np.argmin(yE)])
#    
################################################################################################################

# 1.	Intializaiton of the vertices
# 2.	Construct the Delaunay triangulations of S
# 3.	Constrict an appropriate regression model of S. calculate the search function as sc(x) = p(x)-Ke(x).
#  evaluate the discrete search funciton as:
#           sd = min ( p(x) , yi + (yi-p(x))) - L * sigma(h,T) - sigma(Lx)
# 4.	If sc <= sd then evaluate the minimizer of the continuous search function.
# 5.       else:
#                calculate the loss function as:
#                    minimize_{N,T}     Loss = [ N log(N) ]^n * T * N
#                    such that          L * sigma(h,T)   <   eps
#                       where eps = min { (sd_i - sc),  (sd_i - sd{i-1}) }   [with minimum N value]
#           Improve the existing point accuracy with the N_new and T_new.

# data = {'xE': xE, 'xU': xU, 'yE': yE, 'sigma': SigmaT, 'T': T}
# io.savemat("schwefel_2D", data)
