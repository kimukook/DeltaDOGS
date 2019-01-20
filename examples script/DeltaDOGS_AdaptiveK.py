import numpy as np
import scipy.io
from functools import partial
from dogs import interpolation
from dogs import Utils
from dogs import adaptiveK
from dogs import cartesian_grid
from dogs import Dogsplot

float_formatter = lambda x: "%.4f" % x
# only print 4 digits
np.set_printoptions(formatter={'float_kind': float_formatter})

# This script shows the Delta DOGS(Lambda) Adaptive K main code


n = 1              # Dimenstion of input data
fun_arg = 2        # Type of function evaluation
iter_max = 100     # Maximum number of iterations based on each mesh
MeshSize = 8       # Represents the number of mesh refinement that algorithm will perform


# plot class
plot_index = 1
original_plot = 0
illustration_plot = 0
interpolation_plot = 0
subplot_plot = 0
store_plot = 1     # The indicator to store ploting results as png.
nff = 1  # Number of experiments

# Algorithm choice:
sc = "AdaptiveK"   # The type of continuous search function
alg_name = 'DDOGS/'
# Determines which optimization solver to use.
snopt_solver = 1          # Indicate the optimization package that we use. snopt = 1, using snopt.
dnopt_solver = 0          # Indicate the optimization package that we use. dnopt = 1, using dnopt.
if snopt_solver == 1:
    from dogs import adaptiveK_snopt
if dnopt_solver == 1:
    from dogs import adaptiveK_dnopt

# Calculate the Initial trinagulation points
num_iter = 0       # Represents how many iteration the algorithm goes
Nm = 8             # Initial mesh grid size
L_refine = 0       # Initial refinement sign

# Truth function
fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
func_eval = partial(Utils.fun_eval, fun, lb, ub)

xU = Utils.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)
Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

regret = np.zeros((nff, iter_max))
estimate = np.zeros((nff, iter_max))
datalength = np.zeros((nff, iter_max))
mesh = np.zeros((nff, iter_max))

for ff in range(nff):
    newadd = []                  # Identify the point already exists in xU or not.
    if fun_arg != 4:
        if n == 1:
            xE = np.array([[0, 1]])
        elif n == 2:
            xE = np.array([[0.875, 0.5, 0.25, 0.125, 0.75, 0.5], [0.25, 0.125, 0.875, 0.5, 0.875, 0.75]])
        else:
            num_ini = 5 * n
            xE = Utils.random_initial(n, num_ini, Nm)
        num_ini = xE.shape[1]

        yE = np.zeros(xE.shape[1])
        
        # Calculate the function at initial points
        for ii in range(xE.shape[1]):
            yE[ii] = func_eval(xE[:, ii])
    if plot_index:
        plot_parameters = {'original': original_plot, 'illustration': illustration_plot,
                           'subplot': subplot_plot, 'interpolation': interpolation_plot,
                           'store': store_plot, 'xU_plot': xU}
        plot_class = Dogsplot.plot(plot_parameters, sc, num_ini, ff, fun_arg, alg_name)
    inter_par = interpolation.Inter_par(method="NPS")

    for kk in range(MeshSize):
        # if num_iter == 0:
        #     break
        for k in range(iter_max):
            #     break
            num_iter += 1

            K0 = np.ptp(yE, axis=0)  # scale the domain

            [inter_par, yp] = interpolation.interpolateparameterization(xE, yE, inter_par)

            if (num_iter == 1 or num_iter == 9) and n == 1:
                print('z')
                Dogsplot.sc_interp_1D_separate_delaunay_adaptivek(inter_par, xE, yE, fun_arg)

            ypmin = np.amin(yp)
            ind_min = np.argmin(yp)

            # Calcuate the unevaluated function:
            yu = np.zeros([1, xU.shape[1]])
            if xU.shape[1] != 0:
                for ii in range(xU.shape[1]):
                    yu[0, ii] = (interpolation.interpolate_val(xU[:, ii], inter_par) - y0) / Utils.mindis(xU[:, ii], xE)[0]
            else:
                yu = np.array([[]])
            if xU.shape[1] != 0 and n * np.amin(yu) < y0:
                t = np.amin(yu)
                ind = np.argmin(yu)
                xc_eval = np.copy(xU[:, ind].reshape(-1, 1))
                yc = -np.inf
                xU = scipy.delete(xU, ind, 1)  # create empty array
            else:
                while 1:
                    # minimize s_c
                    xs, ind_min = cartesian_grid.add_sup(xE, xU, ind_min)

                    if dnopt_solver == 1:
                        xc, yc, result, func = adaptiveK_dnopt.trisearch_bound_dnopt(inter_par, xs, y0, K0, ind_min)

                    elif snopt_solver == 1:
                        xc, yc = adaptiveK_snopt.triangulation_search_bound_snopt(inter_par, xs, y0, K0, ind_min)

                    else:
                        xc, yc, result, func = adaptiveK.tringulation_search_bound(inter_par, xs, y0, K0, ind_min)

                    if interpolation.interpolate_val(xc, inter_par) < min(yp):
                        xc_grid = np.round(xc * Nm) / Nm
                        xc_eval = np.copy(xc_grid)
                        break
                    else:
                        xc_grid = np.round(xc * Nm) / Nm
                        if Utils.mindis(xc_grid, xE)[0] < 1e-6:
                            break
                        xc_grid, xE, xU, success, newadd = cartesian_grid.points_neighbers_find(xc_grid, xE, xU, Bin, Ain)
                        if success == 1:
                            break
                        else:
                            yu = np.hstack([yu, (interpolation.interpolate_val(xc_grid, inter_par) - y0)
                                            / Utils.mindis(xc_grid, xE)[0]])
                # The following block represents inactivate step: Shahrouz phd thesis P148.
                if Utils.mindis(xc_grid, xE)[0] < 1e-6:  # xc_grid already exists, mesh refine.
                    L_refine = 1  # paper criteria (4)

                else:
                    if newadd == 0:  # inactivate step: xc_grid already exists in xU
                        # paper criteria (1) (b)
                        xc_eval = np.copy(xc_grid)
                        newadd = []  # Point found from xU that already exists, delete that point in xU
                        dis, index, xu = Utils.mindis(xc_eval, xU)
                        xU = scipy.delete(xU, index, 1)

                    else:
                        sd_xc = (interpolation.interpolate_val(xc, inter_par) - y0) / Utils.mindis(xc, xE)[0]
                        if xU.shape[1] != 0 and sd_xc > n * np.amin(yu):
                            # paper criteria (2)
                            ind = np.argmin(yu)
                            xc_eval = np.copy(xU[:, ind].reshape(-1, 1))
                            yc = -np.inf
                            xU = scipy.delete(xU, ind, 1)
                                
                        else:  # sd_xc < np.amin(yu) 
                                # paper criteria (3)
                            xc_eval = xc_grid
                            
            if L_refine == 1:
                Nm *= 2 
                L_refine = 0
                print('===============  MESH Refinement  ===================')
                summary = {'alg': 'ori', 'xc_grid': xc_grid, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
                Dogsplot.print_summary(num_iter, k, kk, xE, yE, summary)
                if plot_index and n == 2:
                # Only work for 2D
                    plot_class = Dogsplot.plot_delta_dogs(xE, Nm, plot_class, alg_name)
                break
                # For 1D test, use Dogsplot.sc_interp_1D_separate_delaunay()
            else:
                xE = np.hstack((xE, xc_eval))
                yE = np.hstack((yE, func_eval(xc_eval)))
            try:
                summary = {'alg': 'ori', 'xc_grid': xc_grid, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
                Dogsplot.print_summary(num_iter, k, kk, xE, yE, summary)
                if plot_index and n == 2:
                    # Only work f or 2D
                    plot_class = Dogsplot.plot_delta_dogs(xE, Nm, plot_class, alg_name)
            except:
                print('Support points')

    Alg = {'name': alg_name}
    Dogsplot.save_data(xE, yE, inter_par, ff, Alg)
    Dogsplot.save_results_txt(xE, yE, fname, y0, xmin, Nm, Alg, ff, num_ini)
    Dogsplot.dogs_summary_plot(xE, yE, y0, ff, xmin, fname, alg_name)
