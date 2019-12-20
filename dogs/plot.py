import  os
import  matplotlib.pyplot   as plt
import  numpy               as np
from    dogs                import Utils
from    dogs                import cartesian_grid
import  scipy.io            as io
from    matplotlib.ticker   import PercentFormatter
from    scipy.spatial       import Delaunay
from    mpl_toolkits.mplot3d import axes3d, Axes3D


class PlotClass:
    def __init__(self):
        self.size = 100

    def fig_saver(self, name, ddogs):
        if ddogs.save_fig:
            name = os.path.join(ddogs.plot_folder, name) + '.' + ddogs.fig_format
            plt.savefig(name, format=ddogs.fig_format, dpi=200)
        else:
            pass


    def initial_calc1D(self, ddogs):
        """
        Calculate the initial objective function in 1D parameter space.
        :return:
        """
        ddogs.func_prior_xE = np.linspace(ddogs.physical_lb[0], ddogs.physical_ub[0], self.size)
        ddogs.func_prior_yE = np.zeros(ddogs.func_prior_xE.shape[0])
        for i in range(self.size):
            ddogs.func_prior_yE[i] = ddogs.func_eval(ddogs.func_prior_xE[i])

        # Define the range of objective function
        if not ddogs.func_range_prior:
            ddogs.plot_ylow = np.min(ddogs.func_prior_yE) - 1
            ddogs.plot_yupp = np.max(ddogs.func_prior_yE) + 1


    def initial_calc2D(self, ddogs):
        """
        Calculate the initial objective function in 2D parameter space.
        :return:
        """
        x = np.linspace(ddogs.physical_lb[0], ddogs.physical_ub[0], self.size)
        y = np.linspace(ddogs.physical_lb[1], ddogs.physical_ub[1], self.size)
        ddogs.func_prior_xE_2DX, ddogs.func_prior_xE_2DY = np.meshgrid(x, y)
        ddogs.func_prior_xE_2DZ = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                point = np.array([[ddogs.func_prior_xE_2DX[i, j]], [ddogs.func_prior_xE_2DY[i, j]]])
                ddogs.func_prior_xE_2DZ[i, j] = ddogs.func_eval(point)

    def initial_plot1D(self, ddogs):
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor((0.773, 0.769, 0.769))

        # Display the range of the plot
        plt.ylim(ddogs.plot_ylow, ddogs.plot_yupp)
        plt.xlim(ddogs.physical_lb - .05, ddogs.physical_ub + .05)

        # plot the objective function
        plt.plot(ddogs.func_prior_xE, ddogs.func_prior_yE, 'k', label=r'$f(x)$')

        plt.grid(color='white')
        plt.legend(loc='lower left')
        self.fig_saver('initial1D', ddogs)
        plt.close(fig)

    def initial_plot2D(self, ddogs):
        fig = plt.figure(figsize=[16, 9])
        ax = Axes3D(fig)
        ax.set_facecolor((0.773, 0.769, 0.769))
        ax.plot_wireframe(ddogs.func_prior_xE_2DX, ddogs.func_prior_xE_2DY, ddogs.func_prior_xE_2DZ, alpha=.5, color='g')
        self.fig_saver('initial1D', ddogs)
        plt.close(fig)


    def plot1D(self, ddogs):
        """
        1D iteration information ploter:
        1) Every additional sampling, update the uncertainty
        2) Every identifying sampling, plot the minimizer of continuous search function

        :return:
        """
        fig = plt.figure(figsize=[16, 9])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor((0.773, 0.769, 0.769))

        if ddogs.iter_type == 'scmin':
            plot_xE = np.copy(ddogs.xE[:, :-1])
            plot_yE = np.copy(ddogs.yE[:-1])
        else:
            plot_xE = np.copy(ddogs.xE)
            plot_yE = np.copy(ddogs.yE)

        # scatter plot the evaluated points
        plt.scatter(plot_xE, plot_yE, marker='s', c='b')

        # plot the objective function
        plt.plot(ddogs.func_prior_xE, ddogs.func_prior_yE, 'k')

        # plot the continuous search function
        self.continuous_search_plot1D(ddogs)

        # Display the range of the plot
        plt.ylim(ddogs.plot_ylow, ddogs.plot_yupp)
        plt.xlim(ddogs.physical_lb - .05, ddogs.physical_ub + .05)

        plt.grid(color='white')
        plt.legend(loc='lower left')
        self.fig_saver('plot1D' + str(ddogs.iter), ddogs)
        plt.close(fig)

    def continuous_search_plot1D(self, ddogs):
        """
        Plot the continuous search function in 1D parameter space.
        :return:
        """
        xU = Utils.bounds(ddogs.lb, ddogs.ub, ddogs.n)
        if ddogs.iter_type == 'scmin':
            xi, _ = cartesian_grid.add_sup(ddogs.xE[:, :-1], xU, 0)
        else:
            xi, _ = cartesian_grid.add_sup(ddogs.xE, xU, 0)

        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)

        num_plot_points = 500
        xe_plot = np.zeros((tri.shape[0], num_plot_points))
        e_plot = np.zeros((tri.shape[0], num_plot_points))
        sc_plot = np.zeros((tri.shape[0], num_plot_points))

        for ii in range(len(tri)):
            simplex_range = np.copy(xi[:, tri[ii, :]])

            # Discretized mesh grid on x direction in one simplex
            x = np.linspace(simplex_range[0, 0], simplex_range[0, 1], num_plot_points)

            # Circumradius and circumcenter for current simplex
            R2, xc = Utils.circhyp(xi[:, tri[ii, :]], ddogs.n)

            for jj in range(len(x)):
                # Interpolation p(x)
                p = ddogs.inter_par.inter_val(x[jj])

                # Uncertainty function e(x)
                e_plot[ii, jj] = (R2 - np.linalg.norm(x[jj] - xc) ** 2)

                # Continuous search function s(x)
                sc_plot[ii, jj] = p - ddogs.K * ddogs.K0 * e_plot[ii, jj]

            xe_plot[ii, :] = np.copy(x)

        for i in range(len(tri)):
            # Plot the uncertainty function e(x)
            plt.plot(xe_plot[i, :], e_plot[i, :] - np.abs(ddogs.plot_ylow), c='g', label=r'$e(x)$')
            # Plot the continuous search function sc(x)
            plt.plot(xe_plot[i, :], sc_plot[i, :], 'r--', zorder=20, label=r'$S_c(x)$')

        yc_min = sc_plot.flat[np.abs(xe_plot - ddogs.xc).argmin()]
        plt.scatter(ddogs.xc, yc_min, c=(1, 0.769, 0.122), marker='D', zorder=15, label=r'min $S_c(x)$')

    def plot2D(self, ddogs):
        fig = plt.figure(figsize=[16, 9])
        ax = Axes3D(fig)
        # calculate the continuous search function over the entire domain
        continuous_search_Z = np.zeros(ddogs.func_prior_xE_2DX.shape)
        xU = Utils.bounds(ddogs.lb, ddogs.ub, ddogs.n)
        if ddogs.iter_type == 'scmin':
            xi, _ = cartesian_grid.add_sup(ddogs.xE[:, :-1], xU, 0)
        else:
            xi, _ = cartesian_grid.add_sup(ddogs.xE, xU, 0)

        options = 'Qt Qbb Qc' if ddogs.n <= 3 else 'Qt Qbb Qc Qx'
        DT = Delaunay(xi.T, qhull_options=options)

        for i in range(self.size):
            for j in range(self.size):
                point = np.array([[ddogs.func_prior_xE_2DX[i, j], ddogs.func_prior_xE_2DY[i, j]]])
                index = DT.find_simplex(point)
                simplex = xi[:, DT.simplices[index][0]]
                R2, xc = Utils.circhyp(simplex, ddogs.n)

                p = ddogs.inter_par.inter_val(point)
                e = R2 - np.linalg.norm(point - xc) ** 2
                continuous_search_Z[i, j] = p - ddogs.K * ddogs.K0 * e
        # plot the objective and continuous search
        ax.plot_wireframe(ddogs.func_prior_xE_2DX, ddogs.func_prior_xE_2DY, ddogs.func_prior_xE_2DZ,
                          alpha=0.5, color='g', label=r'$f(x)$')
        ax.plot_wireframe(ddogs.func_prior_xE_2DX, ddogs.func_prior_xE_2DY, continuous_search_Z,
                          alpha=0.5, color='b', label=r'$s(x)$')

        # scatter plot the evaluated data points
        if ddogs.iter_type == 'scmin':
            plot_xE = ddogs.xE[0, :-1]
            plot_yE = ddogs.xE[1, :-1]
            plot_zE = ddogs.yE[:-1]
        else:
            plot_xE = ddogs.xE[0, :]
            plot_yE = ddogs.xE[1, :]
            plot_zE = ddogs.yE

        ax.scatter(plot_xE, plot_yE, plot_zE, c='r', marker='o')
        ax.grid(color='white')
        ax.legend(loc='lower left')
        self.fig_saver('plot2D' + str(ddogs.iter), ddogs)
        plt.close(fig)

    def plot2D_contour(self, ddogs):
        fig = plt.figure(figsize=[16, 9])
        l = np.linspace(np.min(ddogs.func_prior_xE_2DZ), np.max(ddogs.func_prior_xE_2DZ), 30)
        plt.contourf(ddogs.func_prior_xE_2DX, ddogs.func_prior_xE_2DY, ddogs.func_prior_xE_2DZ, cmap='gray', levels=l)
        plt.scatter(ddogs.xE[0, :], ddogs.xE[1, :], c='w', marker='s', s=70, edgecolors='k')
        plt.scatter(ddogs.xE[0, -1], ddogs.xE[1, -1], c='r', marker='s', s=50, edgecolors='k')

        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        self.fig_saver('plot2D' + str(ddogs.iter), ddogs)
        plt.close(fig)

    def summary_display(self, ddogs):
        if ddogs.xmin is not None:
            pos_reltv_error = str(np.round(np.linalg.norm(ddogs.xmin - ddogs.xE[:, np.argmin(ddogs.yE)])
                                           / np.linalg.norm(ddogs.xmin) * 100, decimals=4)) + '%'
            val_reltv_error = str(np.round(np.abs(np.min(ddogs.yE) - ddogs.y0)
                                           / np.abs(ddogs.y0) * 100, decimals=4)) + '%'
        else:
            pos_reltv_error = 0
            val_reltv_error = 0

        if ddogs.y0 is not None:
            cur_pos_reltv_err = str(np.round(np.linalg.norm(ddogs.xmin - ddogs.xE[:, -1])
                                             / np.linalg.norm(ddogs.xmin) * 100, decimals=4)) + '%'
            cur_val_reltv_err = str(np.round(np.abs(ddogs.yE[-1] - ddogs.y0) / np.abs(ddogs.y0) * 100,
                                             decimals=4)) + '%'
        else:
            cur_pos_reltv_err = 0
            cur_val_reltv_err = 0

        if ddogs.iter_type == 'sdmin':
            iteration_name = 'Additional sampling'
        elif ddogs.iter_type == 'scmin':
            iteration_name = 'Identifying sampling'
        elif ddogs.iter_type == 'refine':
            iteration_name = 'Mesh refine iteration'
        else:
            iteration_name = 'Bug happens!'
        print('============================   ', iteration_name, '   ============================')
        print(' %40s ' % 'No. Iteration', ' %30s ' % ddogs.iter)
        print(' %40s ' % 'Mesh size', ' %30s ' % ddogs.ms)
        print(' %40s ' % 'X-min', ' %30s ' % ddogs.xmin.T[0])
        print(' %40s ' % 'Target Value', ' %30s ' % ddogs.y0)
        print("\n")
        print(' %40s ' % 'Candidate point', ' %30s ' % ddogs.xE[:, np.argmin(ddogs.yE)])
        print(' %40s ' % 'Candidate FuncValue', ' %30s ' % np.min(ddogs.yE))
        print(' %40s ' % 'CandidatePosition RelativeError', ' %30s ' % pos_reltv_error)
        print(' %40s ' % 'CandidateValue RelativeError', ' %30s ' % val_reltv_error)
        print("\n")
        print(' %40s ' % 'Current point', ' %30s ' % ddogs.xE[:, -1])
        print(' %40s ' % 'Current FuncValue', ' %30s ' % ddogs.yE[-1])
        print(' %40s ' % 'Current Position RelativeError', ' %30s ' % cur_pos_reltv_err)
        print(' %40s ' % 'Current Value RelativeError', ' %30s ' % cur_val_reltv_err)
        if not ddogs.iter_type == 'refine':
            print(' %40s ' % 'CurrentEval point', ' %30s ' % ddogs.xE[:, -1])
            print(' %40s ' % 'FuncValue', ' %30s ' % ddogs.yE[-1])
        print("\n")

    def summary_plot(self, ddogs):
        """
        This function generates the summary information of Deltddogs optimization
        :param yE:  The function values evaluated at each iteration
        :param y0:  The target minimum of objective function.
        :param folder: Identify the folder we want to save plots. "DDOGS" or "DimRed".
        :param xmin: The global minimizer of test function, usually presented in row vector form.
        :param ff:  The number of trial.
        """
        N = ddogs.yE.shape[0]  # number of iteration
        if ddogs.y0 is not None:
            yE_best = np.zeros(N)
            yE_reltv_error = np.zeros(N)
            for i in range(N):
                yE_best[i] = min(ddogs.yE[:i + 1])
                yE_reltv_error[i] = (np.min(ddogs.yE[:i + 1]) - ddogs.y0) / np.abs(ddogs.y0) * 100
            # Plot the function value of candidate point for each iteration
            fig, ax1 = plt.subplots()
            plt.grid()
            # The x-axis is the function count, and the y-axis is the smallest value DELTA-DOGS had found.
            ax1.plot(np.arange(N) + 1, yE_best, label='Function value of Candidate point', c='b')
            ax1.plot(np.arange(N) + 1, ddogs.y0 * np.ones(N), label='Global Minimum', c='r')
            ax1.set_ylabel('Function value', color='b')
            ax1.tick_params('y', colors='b')
            plt.xlabel('Number of Evaluated Datapoints')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

            # Plot the relative error on the right twin-axis.
            ax2 = ax1.twinx()
            ax2.plot(np.arange(N) + 1, yE_reltv_error, 'g--', label=r'Relative Error=$\frac{f_{min}-f_{0}}{|f_{0}|}$')
            ax2.set_ylabel('Relative Error', color='g')

            ax2.yaxis.set_major_formatter(PercentFormatter())
            ax2.tick_params('y', colors='g')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
            # Save the plot
            self.fig_saver('Candidate_point', ddogs)
            plt.close(fig)
        else:
            print('Target value y0 is not available, no candidate plot saved.')
        ####################   Plot the distance of candidate x to xmin of each iteration  ##################
        if ddogs.xmin is not None:
            fig2 = plt.figure()
            plt.grid()
            xE_dis = np.zeros(N)
            for i in range(N):
                index = np.argmin(ddogs.yE[:i + 1])
                xE_dis[i] = np.linalg.norm(ddogs.xE[:, index].reshape(-1, 1) - ddogs.xmin)
            plt.plot(np.arange(N) + 1, xE_dis, label="Distance with global minimizer")
            plt.ylabel('Distance value')
            plt.xlabel('Number of Evaluated Datapoints')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
            self.fig_saver('Distance', ddogs)
            plt.close(fig2)
        else:
            print('Global minimizer xmin is not available, no candidate plot saved.')

    def result_saver(self, ddogs):
        ddogs_data = {}
        ddogs_data['xE'] = ddogs.xE
        ddogs_data['yE'] = ddogs.yE
        if ddogs.inter_par is not None:
            ddogs_data['inter_par_method'] = ddogs.inter_par.method
            ddogs_data['inter_par_w'] = ddogs.inter_par.w
            ddogs_data['inter_par_v'] = ddogs.inter_par.v
            ddogs_data['inter_par_xi'] = ddogs.inter_par.xi
        name = os.path.join(ddogs.plot_folder, 'data.mat')
        io.savemat(name, ddogs_data)

    @staticmethod
    def result_reader(name):
        data = io.loadmat(name)
        xE = data['xE']
        yE = data['yE']
        return xE, yE
