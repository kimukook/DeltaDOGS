"""
//
//  deltaDOGS.py
//  Delta-DOGS:  Delaunay-based derivative-free optimization via global surrogate (∆-DOGS) is a
//               highly-efficient modern variant of Response Surface Method(RSM) that, under the
//               appropriate assumptions, guarantees convergence to the global solution for nonconvex
//               optimization problems with computationally expensive objective function.
//
//  Created by Muhan Zhao on 12/4/19.
//  Copyright © 2019 Muhan Zhao. All rights reserved.
//
"""
import  scipy
import  os
import  inspect
import  shutil
import  numpy               as np
from    functools           import partial
from    dogs                import Utils
from    dogs                import interpolation
from    dogs                import cartesian_grid
from    dogs                import constantK
from    dogs                import adaptiveK
from    dogs                import plot
inf = 1e+20


class OptionsClass:
    """
    Options Class
    """

    def __init__(self):
        self.options = None
        self.solverName = 'None'

    def set_option(self, key, value):
        try:
            if type(value) is self.options[key][2]:
                self.options[key][0] = value
            else:
                print(f"The type of value for the keyword '{key}' should be '{self.options[key][2]}'.")
        except:
            raise ValueError('Incorrect option keyword or type: ' + key)

    def get_option(self, key):
        try:
            value = self.options[key][0]
            return value
        except:
            raise ValueError('Incorrect option keyword: ' + key)

    def reset_options(self, key):
        try:
            self.options[key] = self.options[key][1]
        except:
            raise ValueError('Incorrect option keyword: ' + key)


class DeltaDOGSOptions(OptionsClass):
    """
    Options class for alphaDOGS
    """
    def __init__(self):
        OptionsClass.__init__(self)
        self.setup()
        self.solverName = 'DeltaDOGS'

    def setup(self):
        self.options = {
            # [Current value, default value, type]
            'Objective function name'           : [None, None, str],
            'Constant surrogate'                : [False, False, bool],
            'Constant K'                        : [6.0, 6.0, float],

            'Adaptive surrogate'                : [False, False, bool],
            'Target value'                      : [None, None, float],

            'Scipy solver'                      : [False, False, bool],
            'Snopt solver'                      : [False, False, bool],

            'Initial mesh size'                 : [3, 3, int],
            'Number of mesh refinement'         : [8, 8, int],

            'Initial sites known'               : [False, False, bool],
            'Initial sites'                     : [None, None, np.ndarray],
            'Initial function values'           : [None, None, np.ndarray],

            'Global minimizer known'            : [False, False, bool],
            'Global minimizer'                  : [None, None, np.ndarray],

            'Function range known'              : [False, False, bool],
            'Function range'                    : [None, None, np.ndarray],

            'Function evaluation cheap'         : [True, True, bool],
            'Function prior file path'          : [None, None, str],
            'Plot saver'                        : [True, True, bool],
            'Figure format'                     : ['png', 'png', str],
            'Candidate distance summary'        : [False, False, bool],
            'Candidate objective value summary' : [False, False, bool],
            'Iteration summary'                 : [False, False, bool],
            'Optimization summary'              : [False, False, bool]
        }


class DeltaDOGS:
    def __init__(self, bounds, func_eval, options, A=None, b=None):
        """

        :param bounds:
        :param func_eval:
        :param A:
        :param b:
        :param options:
        """
        # n: The dimension of input parameter space
        self.n = bounds.shape[0]
        # physical lb & ub: Normalize the physical bounds
        self.physical_lb = bounds[:, 0].reshape(-1, 1)
        self.physical_ub = bounds[:, 1].reshape(-1, 1)
        # lb & ub: The normalized search bounds
        self.lb = np.zeros((self.n, 1))
        self.ub = np.ones((self.n, 1))

        # ms: The mesh size for each iteration. This is the initial definition.
        self.initial_mesh_size = options.get_option('Initial mesh size')
        self.ms = 2 ** self.initial_mesh_size
        self.num_mesh_refine = options.get_option('Number of mesh refinement')
        self.max_mesh_size = 2 ** (self.initial_mesh_size + self.num_mesh_refine)

        self.iter = 0

        # Define the surrogate model
        if options.get_option('Constant surrogate') and options.get_option('Adaptive surrogate'):
            raise ValueError('Constant and Adaptive surrogate both activated. Set one to False then rerun code.')
        elif not options.get_option('Constant surrogate') and not options.get_option('Adaptive surrogate'):
            raise ValueError('Constant and Adaptive surrogate both inactivated. Set one to True then rerun code.')
        elif options.get_option('Constant surrogate'):
            self.surrogate_type = 'c'
        elif options.get_option('Adaptive surrogate'):
            self.surrogate_type = 'a'
        else:
            pass

        if self.surrogate_type == 'c':
            # Define the parameters for discrete and continuous constant search function
            self.K = options.get_option('Constant K')
        elif self.surrogate_type == 'a':
            if options.get_option('Target value') is None:
                print('Target value y0 is None type while applying adaptive surrogate.')
            else:
                self.y0 = options.get_option('Target value')

        # Define the linear constraints, Ax <= b. if A and b are None type, set them to be the box domain constraints.
        if (A and b) is None:
            self.Ain = np.concatenate((np.identity(self.n), -np.identity(self.n)), axis=0)
            self.Bin = np.concatenate((np.ones((self.n, 1)), np.zeros((self.n, 1))), axis=0)
        else:
            pass

        if options.get_option('Scipy solver') and options.get_option('Snopt solver'):
            raise ValueError('More than one optimization solver specified, check Scipy solver or Snopt solver')
        elif not options.get_option('Scipy solver') and not options.get_option('Snopt solver'):
            raise ValueError('No optimization solver specified.')
        elif options.get_option('Scipy solver'):
            self.solver_type = 'scipy'
        elif options.get_option('Snopt solver'):
            self.solver_type = 'snopy'
        else:
            pass

        # Initialize the function evaluation, capsulate those three functions with physical bounds
        self.func_eval = partial(Utils.fun_eval, func_eval, self.physical_lb, self.physical_ub)

        # Define the global optimum and its values
        if options.get_option('Global minimizer known'):
            self.xmin = Utils.normalize_bounds(options.get_option('Global minimizer'), self.physical_lb,
                                               self.physical_ub)
            self.y0 = options.get_option('Target value')
        else:
            self.xmin = None
            self.y0 = None

        # Define the iteration type for each sampling iteration.
        self.iter_type = None

        # Define the initial sites and their function evaluations
        if options.get_option('Initial sites known'):
            physical_initial_sites = options.get_option('Initial sites')
            # Normalize the bound
            self.xE = Utils.normalize_bounds(physical_initial_sites, self.physical_lb, self.physical_ub)
        else:
            self.xE = Utils.random_initial(self.n, 2 * self.n, self.ms, self.Ain, self.Bin)

        if options.get_option('Initial function values') is not None:
            self.yE = options.get_option('Initial function values')
        else:
            # Compute the function values, time length, and noise level at initial sites
            self.yE = np.zeros(self.xE.shape[1])

            for i in range(2 * self.n):
                self.yE[i] = self.func_eval(self.xE[:, i])

        # Define the initial support points
        self.xU = Utils.bounds(self.lb, self.ub, self.n)
        self.xU = Utils.unique_support_points(self.xU, self.xE)
        self.yu = None

        self.K0 = np.ptp(self.yE, axis=0)
        # Define the interpolation
        self.inter_par = None
        self.yp = None

        # Define the discrete search function
        self.sd = None

        # Define the minimizer of continuous search function, parameter to be evaluated, xc & yc.
        self.xc = None
        self.yc = None

        # Define the minimizer of discrete search function
        self.xd = None
        self.yd = None
        self.index_min_yd = None
        self.newadd = None

        # Define the name of directory to store the figures
        self.algorithm_name = options.solverName

        #  ==== Plot section here ====
        self.plot = plot.PlotClass()
        # Define the option whether or not to save the optimization result for each iteration
        self.save_fig = options.get_option('Plot saver')
        # Define the figure format
        self.fig_format = options.get_option('Figure format')

        # Generate the folder path
        self.func_path = options.get_option('Objective function name')  # Function folder, e.g. Lorenz
        # Directory path
        self.current_path = None
        # Figures folder
        self.plot_folder = None
        self.folder_path_generator()
        # Define the name of the function called.
        self.func_name = options.get_option('Objective function name')

        self.func_prior_xE = None
        self.func_prior_yE = None
        self.func_prior_xE_2DX = None
        self.func_prior_xE_2DY = None
        self.func_prior_xE_2DZ = None

        if not options.get_option('Function evaluation cheap'):
            if self.save_fig:
                print('Function evaluation is expensive, turn figure saving option to negative.')
                self.save_fig = False

        self.func_range_prior = options.get_option('Function range known')
        if options.get_option('Function evaluation cheap') and self.save_fig:
            if self.n == 1:
                self.plot.initial_calc1D(self)
                self.initial_plot = self.plot.initial_plot1D
                self.iter_plot = self.plot.plot1D

            elif self.n == 2:
                self.plot.initial_calc2D(self)
                self.initial_plot = self.plot.initial_plot2D
                self.iter_plot = self.plot.plot2D_contour

            elif self.n >= 3:
                print('Dimension of parameter >= 3, turn figure saving option to negative.')
                self.save_fig = False

                self.plot_ylow = None
                self.plot_yupp = None
            else:
                pass

        if self.save_fig:
            if self.func_range_prior:
                self.plot_ylow = options.get_option('Function range')[0]
                self.plot_yupp = options.get_option('Function range')[1]
            elif self.n > 3:
                self.plot_ylow = -2 * np.max(np.abs(self.yE))
                self.plot_yupp = 2 * np.max(np.abs(self.yE))
        else:
            pass

        self.iter_summary = options.get_option('Iteration summary')
        self.optm_summary = options.get_option('Optimization summary')

    def folder_path_generator(self):
        '''
        Determine the root path and generate the plot folder.
        :return:
        '''
        self.current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.plot_folder = os.path.join(self.current_path, 'plot')
        if os.path.exists(self.plot_folder):
            shutil.rmtree(self.plot_folder)
        os.makedirs(self.plot_folder)

    def deltadogs_optimizer(self):
        """
        Main optimization function.
        :return:
        """
        for kk in range(self.num_mesh_refine):
            for k in range(20):
                if self.surrogate_type == 'c':
                    self.constant_surrogate_solver()
                else:
                    self.adaptive_surrogate_solver()

                if self.iter_type == 'refine':
                    self.iter_type = None
                    break
                else:
                    pass
        xmin = self.xE[:, np.argmin(self.yE)]
        if self.optm_summary and self.save_fig:
            self.plot.summary_plot(self)
        self.plot.result_saver(self)
        return xmin

    def constant_surrogate_solver(self):
        self.iter += 1
        self.K0 = np.ptp(self.yE, axis=0)
        self.inter_par = interpolation.InterParams(self.xE)
        self.yp = self.inter_par.interpolateparameterization(self.yE)

        ind_min = np.argmin(self.yp)

        yu = np.zeros(self.xU.shape[1])
        if self.xU.shape[1] > 0:
            for ii in range(self.xU.shape[1]):
                yu[ii] = self.inter_par.inter_val(self.xU[:, ii]) - \
                         self.K * self.K0 * Utils.mindis(self.xU[:, ii], self.xE)[0]
        else:
            yu = inf

        if self.xU.shape[1] > 0 and np.min(yu) < np.min(self.yE):
            ind = np.argmin(yu)
            self.xc = np.copy(self.xU[:, ind].reshape(-1, 1))
            self.yc = -inf
            self.xU = scipy.delete(self.xU, ind, 1)
            self.iter_type = 'scmin'
        else:
            while 1:
                xs, ind_min = cartesian_grid.add_sup(self.xE, self.xU, ind_min)
                xc, yc, result = constantK.tringulation_search_bound_constantK(self.inter_par, xs, self.K * self.K0,
                                                                               ind_min)
                xc_grid = np.round(xc * self.ms) / self.ms
                xE, xU, success, self.newadd = cartesian_grid.points_neighbers_find(xc_grid, self.xE, self.xU,
                                                                                    self.Bin, self.Ain)

                if success == 0:  # inactivate step
                    self.xU = np.hstack((self.xU, xc_grid))
                    self.yu = np.hstack((yu, self.inter_par.inter_val(xc_grid)[0] - self.K * self.K0 *
                                         Utils.mindis(xc_grid, self.xE)[0]))

                else:  # activate step
                    sd_xc = self.inter_par.inter_val(xc_grid) - self.K * self.K0 * Utils.mindis(xc_grid, self.xE)[0]
                    if self.xU.shape[1] == 0 or sd_xc < np.min(yu):
                        # xc_grid has lower sd(x) value
                        if Utils.mindis(xc_grid, self.xE)[0] < 1e-6:
                            # point already exist
                            self.iter_type = 'refine'
                            break
                        else:
                            self.iter_type = 'scmin'
                            self.xc = np.copy(xc_grid)
                            break

                    else:
                        # the discrete search function values of the minimizer of sd(x) is lower
                        self.iter_type = 'sdmin'
                        index = np.argmin(yu)
                        self.xc = np.copy(self.xU[:, index].reshape(-1, 1))
                        self.xU = scipy.delete(self.xU, index, 1)
                        break

        if self.iter_type == 'refine':
            self.K *= 2
            self.ms *= 2
        else:
            self.xE = np.hstack((self.xE, self.xc))
            self.yE = np.hstack((self.yE, self.func_eval(self.xc)))

        if self.save_fig:
            self.iter_plot(self)
        if self.iter_summary:
            self.plot.summary_display(self)

    def adaptive_surrogate_solver(self):
        self.iter += 1
        self.K0 = np.ptp(self.yE, axis=0)
        self.inter_par = interpolation.InterParams(self.xE)
        self.yp = self.inter_par.interpolateparameterization(self.yE)

        ind_min = np.argmin(self.yp)

        yu = np.zeros(self.xU.shape[1])
        if self.xU.shape[1] > 0:
            for ii in range(self.xU.shape[1]):
                yu[ii] = (self.inter_par.inter_val(self.xU[:, ii]) - self.y0) / \
                        Utils.mindis(self.xU[:, ii], self.xE)[0]

        if self.xU.shape[1] > 0 and np.min(yu) < np.min(self.yE):
            ind = np.argmin(yu)
            self.xc = np.copy(self.xU[:, ind].reshape(-1, 1))
            self.yc = -inf
            self.xU = scipy.delete(self.xU, ind, 1)
        else:
            while 1:
                xs, ind_min = cartesian_grid.add_sup(self.xE, self.xU, ind_min)
                xc, yc, result = adaptiveK.tringulation_search_bound(self.inter_par, xs, self.y0, self.K0, ind_min)
                xc_grid = np.round(xc * self.ms) / self.ms
                xE, xU, success, self.newadd = cartesian_grid.points_neighbers_find(xc_grid, self.xE, self.xU,
                                                                                    self.Bin, self.Ain)

                if success == 0:  # inactivate step
                    self.xU = np.hstack((self.xU, xc_grid))
                    self.yu = np.hstack((yu, (self.inter_par.inter_val(xc_grid)[0] - self.y0)
                                         / Utils.mindis(xc_grid, self.xE)[0]))

                else:  # activate step
                    sd_xc = (self.inter_par.inter_val(xc_grid) - self.y0) / \
                             Utils.mindis(xc_grid, self.xE)[0]
                    if self.xU.shape[1] == 0 or sd_xc < np.min(yu):
                        # xc_grid has lower sd(x) value
                        if Utils.mindis(xc_grid, self.xE)[0] < 1e-6:
                            # point already exist
                            self.iter_type = 'refine'
                            break
                        else:
                            self.iter_type = 'scmin'
                            self.xc = np.copy(xc_grid)
                            break

                    else:
                        # the discrete search function values of the minimizer of sd(x) is lower
                        self.iter_type = 'sdmin'
                        index = np.argmin(yu)
                        self.xc = np.copy(self.xU[:, index].reshape(-1, 1))
                        self.xU = scipy.delete(self.xU, index, 1)
                        break

        if self.iter_type == 'refine':
            self.ms *= 2
        else:
            self.xE = np.hstack((self.xE, self.xc))
            self.yE = np.hstack((self.yE, self.func_eval(self.xc)))

        if self.save_fig:
            self.iter_plot(self)
        if self.iter_summary:
            self.plot.summary_display(self)
