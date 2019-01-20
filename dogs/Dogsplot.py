"""
Created on Tue Oct 31 15:45:35 2017

@author: mousse
"""
import      os
import      inspect
import      shutil
import      scipy
import      numpy               as np
import      matplotlib.pyplot   as plt
from        matplotlib.ticker   import PercentFormatter
from        scipy               import io
from        functools           import partial
from        dogs                import Utils
from        dogs                import interpolation
'''
Dogsplot.py is implemented to generate results for AlphaDOGS and DeltaDOGS containing the following functions:
    dogs_summary                 :   generate summary plots, 
                                           candidate points 
                                           for each iteration
    plot_alpha_dogs              :   generate plots for AlphaDOGS
    plot_delta_dogs              :   generate plots for DeltaDOGS
    plot_detla_dogs_reduced_dim  :   generate plots for Dimension reduction DeltaDOGS
'''
####################################  Plot Initialization  ####################################


class plot:
    def __init__(self, plot_parameters, sc, num_ini, ff, fun_arg, alg_name):
        '''
        :param plot_index:  Generate plot or not
        :param store_plot:  Store plot or not
        :param sc:          Type of continuous search function, can be 'ConstantK' or 'AdaptiveK'.
        :param num_ini:     Number of initial points
        :param ff:          The times of trials.
        :param fun_arg:     Type of truth function.
        :return:    init_comp : The initialization of contour plots is complete
                    itrp_init_comp : The initialization of interpolation is complete
                    type:   The type of continuous search function.

        '''
        if plot_parameters['original']:
            self.init_comp = 0
        else:
            self.init_comp = -1
        if plot_parameters['illustration']:
            self.illustration_init_comp = 0
        else:
            self.illustration_init_comp = -1
        if plot_parameters['subplot']:
            self.subplot_init_comp = 0
        else:
            self.subplot_init_comp = -1

        if plot_parameters['interpolation']:
            self.itrp_init_comp = 0
        else:
            self.itrp_init_comp = -1
        self.store_plot = plot_parameters['store']
        self.type = sc
        self.num_ini = num_ini
        self.ff = ff
        self.fun_arg = fun_arg
        self.xU_plot = plot_parameters['xU_plot']
        # Make experiments plot folder:
        plot_folder = folder_path(alg_name, ff)
        if os.path.exists(plot_folder):
            # if the folder already exists, delete that folder to restart the experiments.
            shutil.rmtree(plot_folder)
        os.makedirs(plot_folder)


############################################ Plot ############################################
def print_summary(num_iter, k, kk, xE, yE, summary, nm=0, mesh_refine=0):
    n = xE.shape[0]
    Nm = summary['Nm']
    xmin = summary['xmin']
    y0 = summary['y0']
    mindis = format(np.linalg.norm(xmin - xE[:, np.argmin(yE)]), '.4f')
    curdis = format(np.linalg.norm(xmin - xE[:, -1]), '.4f')
    Pos_Reltv_error = str(np.round(np.linalg.norm(xmin - xE[:, np.argmin(yE)])/np.linalg.norm(xmin)*100, decimals=4))+'%'
    Val_Reltv_error = str(np.round((np.min(yE)-y0)/np.abs(y0)*100, decimals=4)) + '%'
    iteration_name = ('Mesh refine' if mesh_refine == 1 else 'Exploration')
    print('============================   ', iteration_name, '   ============================')
    if summary['alg'] == "DR":
        xr = summary['xr']
        w = summary['w'] 
        for i in range(w.shape[0]):
            w[i, 0] = format(w[i, 0], '.4f')
        
        print('%5s' % 'Iter', '%4s' % 'k', '%4s' % 'kk', '%4s' % 'Nm', '%10s' % 'feval_min', '%10s' % 'fevalmin_dis',
              '%10s' % 'cur_f', '%10s' % 'curdis', '%15s' % 'Pos_Reltv_Error', '%15s' % 'Val_Reltv_Error')
        
        print('%5s' % num_iter, '%4s' % k, '%4s' % kk, '%4s' % Nm, '%10s' % format(min(yE), '.4f'),
              '%10s' % mindis, '%10s' % format(yE[-1], '.4f'), '%10s' % curdis, '%15s' % Pos_Reltv_error,
              '%15s' % Val_Reltv_error)
        print('%10s' % '1DSearch', '%10s' % 'RD_Mesh')  # , '%20s' % 'ASM')
        print('%10s' % xr.T[0], '%10s' % nm)  # , '%20s' % w.T[0])
    else:
        print('%5s' % 'Iter', '%4s' % 'k', '%4s'%'kk', '%4s' % 'Nm', '%10s' % 'feval_min', '%10s' % 'fevalmin_dis',
              '%10s' % 'curdis', '%10s' % 'cur_f', '%15s' % 'Pos_Reltv_Error', '%15s' % 'Val_Reltv_Error')
        print('%5s' % num_iter, '%4s' % k, '%4s' % kk, '%4s' % Nm, '%10s' % format(min(yE), '.4f'),
              '%10s' % mindis, '%10s'%curdis, '%10s' % format(yE[-1], '.4f'), '%15s' % Pos_Reltv_error,
              '%15s' % Val_Reltv_error)
    if mesh_refine == 0:
        print('%10s' % 'Best x = ', '%10s' % np.round(xE[:, np.argmin(yE)], decimals=5))
        print('%10s' % 'Current x = ', '%10s' % np.round(xE[:, -1], decimals=5))
    print('====================================================================')
    return 


def dogs_summary_plot(xE, yE, y0, ff, xmin, fun_name, alg_name):
    '''
    This function generates the summary information of DeltaDOGS optimization
    :param yE:  The function values evaluated at each iteration
    :param y0:  The target minimum of objective function.
    :param folder: Identify the folder we want to save plots. "DDOGS" or "DimRed".
    :param xmin: The global minimizer of test function, usually presented in row vector form.
    :param ff:  The number of trial.
    '''
    n = xE.shape[0]
    plot_folder = folder_path(alg_name, ff)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    xmin = np.copy(xmin.reshape(-1, 1))
    N = yE.shape[0]  # number of iteration
    yE_best = np.zeros(N)
    yE_reltv_error = np.zeros(N)
    for i in range(N):
        yE_best[i] = min(yE[:i+1])
        yE_reltv_error[i] = (np.min(yE[:i+1]) - y0) / np.abs(y0) * 100
    # Plot the function value of candidate point for each iteration
    fig, ax1 = plt.subplots()
    plt.grid()
    # The x-axis is the function count, and the y-axis is the smallest value DELTA-DOGS had found.
    # plt.title(alg_name[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fun_name + ': The function value of candidate point', y=1.05)
    ax1.plot(np.arange(N) + 1, yE_best, label='Function value of Candidate point', c='b')
    ax1.plot(np.arange(N) + 1, y0*np.ones(N), label='Global Minimum', c='r')
    ax1.set_ylabel('Function value', color='b')
    ax1.tick_params('y', colors='b')
    plt.xlabel('Iteration number')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Plot the relative error on the right twin-axis.
    ax2 = ax1.twinx()
    ax2.plot(np.arange(N) + 1, yE_reltv_error, 'g--', label=r'Relative Error=$\frac{f_{min}-f_{0}}{|f_{0}|}$')
    ax2.set_ylabel('Relative Error', color='g')

    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params('y', colors='g')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
    # Save the plot
    plt.savefig(plot_folder + "/Candidate_point.eps", format='eps', dpi=1000)
    plt.close(fig)
    ####################   Plot the distance of candidate x to xmin of each iteration  ##################
    fig2 = plt.figure()
    plt.grid()
    plt.title(alg_name[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fun_name + r': The distance of candidate point to $x_{min}$', y=1.05)
    xE_dis = np.zeros(N)
    for i in range(N):
        index = np.argmin(yE[:i+1])
        xE_dis[i] = np.linalg.norm(xE[:, index].reshape(-1,1) - xmin)
    plt.plot(np.arange(N) + 1, xE_dis, label="Distance with global minimizer")
    plt.ylabel('Distance value')
    plt.xlabel('Iteration number')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.savefig(plot_folder + "/Distance.eps", format='eps', dpi=1000)
    plt.close(fig2)
    # fig2, ax3 = plt.subplots()
    # plt.grid()
    # plt.title(Alg[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fun_name + r': The distance of candidate point to $x_{min}$', y=1.05)
    # xE_dis = np.zeros(N)
    # for i in range(N):
    #     index = np.argmin(yE[:i+1])
    #     xE_dis[i] = np.linalg.norm(xE[:, index].reshape(-1,1) - xmin)
    # ax3.plot(np.arange(N) + 1, xE_dis, c='b', label="Distance with global minimizer")
    # ax3.set_ylabel('Function value', color='b')
    # ax3.tick_params('y', colors='b')
    #
    # plt.xlabel('Iteration number')
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    #
    # ax4 = ax3.twinx()
    # ax4.plot(np.arange(1, yE.shape[0]+1), np.ones(yE.shape[0]), 'g', label='Dimension of Active Subspace')
    # ax4.set_ylabel('Dimension', color='g')
    # ax4.tick_params('y', colors='g')
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    #
    # plt.savefig(plot_folder + "/Distance.png", format='png', dpi=1000)
    # plt.close(fig2)
    return 


def save_data(xE, yE, inter_par, ff, Alg):
    name = Alg['name']
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/" + name + str(ff)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    dr_data = {}
    dr_data['xE'] = xE
    dr_data['yE'] = yE
    dr_data['inter_par_method'] = inter_par.method
    dr_data['inter_par_w'] = inter_par.w
    dr_data['inter_par_v'] = inter_par.v
    dr_data['inter_par_xi'] = inter_par.xi
    io.savemat(plot_folder + '/data.mat', dr_data)
    return


def save_results_txt(xE, yE, fun_name, y0, xmin, Nm, Alg, ff, num_ini):
    BestError = (np.min(yE) - y0) / np.linalg.norm(y0)
    Error = (yE - y0) / np.linalg.norm(y0)
    if np.min(Error) > 0.01:
        # relative error > 0.01, optimization performance not good.
        idx = '1% accuracy failed'
    else:
        idx = np.min(np.where(Error-0.01 < 0)[0]) - num_ini
    name = Alg['name']
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/" + name + str(ff)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    with open(plot_folder + '/DminOptmResults.txt', 'w') as f:
        f.write('=====  ' + name[:-1] + ': General information report  ' + str(xE.shape[0]) + 'D' + ' ' + fun_name +
                '=====' + '\n')
        f.write('%40s %10s' % ('Number of Dimensions = ', str(xE.shape[0])) + '\n')
        f.write('%40s %10s' % ('Total Number of Function evaluations = ', str(xE.shape[1])) + '\n')
        f.write('%40s %10s' % ('Actual Error when stopped = ', str(np.round(BestError, decimals=6)*100)+'%') + '\n')
        # f.write('%40s %10s' % ('Best minimizer when stopped = ', str))
        f.write('%40s %10s' % ('Mesh size when stopped = ', str(Nm)) + '\n')
        f.write('%40s %10s' % ('Evaluations Required for 1% accuracy = ', str(idx)) + '\n')
    return
############################    Test function calculator    ################################


def function_eval(X, Y, fun_arg):
    n = 2
    if fun_arg == 1:  # GOLDSTEIN-PRICE FUNCTION
        Z = (1+(X+Y+1)**2*(19-4*X+3*X**2-14*Y+6*X*Y+3*Y**2))*\
            (30+(2*X-3*Y)**2*(18-32*X+12*X**2+48*Y-36*X*Y+27*Y**2))
        y0 = 3
        xmin = np.array([0.5, 0.25])

    elif fun_arg == 2:  # Schwefel
        Z = (-np.multiply(X, np.sin(np.sqrt(abs(500 * X)))) - np.multiply(Y, np.sin(np.sqrt(abs(500 * Y))))) / 2
        y0 = - 1.6759 * n
        xmin = 0.8419 * np.ones((n))

    elif fun_arg == 5:  # Schwefel + Quadratic
        Z = - np.multiply(X / 2, np.sin(np.sqrt(abs(500 * X)))) + 10 * (Y - 0.92) ** 2
        y0 = -10
        xmin = np.array([0.89536, 0.94188])

    elif fun_arg == 6:  # Griewank
        Z = 1 + 1 / 4 * ((X - 0.67) ** 2 + (Y - 0.21) ** 2) - np.cos(X) * np.cos(Y / np.sqrt(2))
        y0 = 0.08026
        xmin = np.array([0.21875, 0.09375])

    elif fun_arg == 7:  # Shubert
        s1 = 0
        s2 = 0
        for i in range(5):
            s1 += (i + 1) * np.cos((i + 2) * (X - 0.45) + (i + 1))
            s2 += (i + 1) * np.cos((i + 2) * (Y - 0.45) + (i + 1))
        Z = s1 * s2
        y0 = -32.7533
        xmin = np.array([0.78125, 0.25])

    elif fun_arg == 8:
        a = 1
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        Z = a*(Y - b*X**2 + c*X - r)**2 + s*(1-t)*np.cos(X) + s
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
        y0 = 0.397887
        xmin = np.array([0.5427728, 0.1516667])  # True minimizer np.array([np.pi, 2.275])
        # 3 minimizer
        xmin2 = np.array([0.123893, 0.8183333])  # xmin2 = np.array([-np.pi, 12.275])
        xmin3 = np.array([0.9616520, 0.165])  # xmin3 = np.array([9.42478, 2.475])

    elif fun_arg == 10:
        x = y = np.linspace(-0.05, 4.05, 1000)
        X, Y = np.meshgrid(x, y)

        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
        y0 = 0
        xmin = np.ones(2)

    elif fun_arg == 11:
        Z = 1
        y0 = -3.86278
        xmin = np.array([0.114614, 0.555649, 0.852547])

    elif fun_arg == 12:
        Z = 1
        y0 = -3.32237
        xmin = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    elif fun_arg == 18:
        Z = np.exp(0.7 * X + 0.3 * Y)
        y0 = -3.32237
        xmin = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    return Z, y0, xmin


def folder_path(alg_name, ff):
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/" + alg_name + str(ff)      # -5 comes from the length of '/dogs'
    return plot_folder


def plot_alpha_dogs(xE, xU, yE, SigmaT, xc_min, yc, yd, funr, num_iter, K, L, Nm):
    '''
    This function is set for plotting Alpha-DOGS algorithm on toy problem, e.g. Schwefel function, containing:
    continuous search function, discrete search function, regression function, truth function and function evaluation.
    
    :param xE: Evaluated points.
    :param xU: Support points, useful for building continuous function.
    :param yE: Function evaluation of xE.
    :param SigmaT: Uncertainty on xE.
    :param xc_min: Minimum point of continuous search function at this iteration.
    :param yc: Minimum of continuous search function built by piecewise quadratic model.
    :param yd: Minimum of discrete search function.
    :param funr: Truth function.
    :param num_iter: Number of iteration.
    :param K: Tuning parameter of continuous search function.
    :param L: Tuning parameter of discrete search function.
    :param Nm: Mesh size.
    :return: Plot for each iteration of Alpha-DOGS algorithm, save the fig at 'plot' folder.
    '''
    #####   Generates plots of cs, ds and function evaluation   ######
    n = xE.shape[0]
    K0 = np.ptp(yE, axis=0)
    #####  Plot the truth function  #####
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([-4, 4])
    xall = np.arange(-0.05, 1.01, 0.001)
    yall = np.arange(-0.05, 1.01, 0.001)
    for i in range(len(xall)):
        yall[i] = funr(np.array([xall[i]]))
    plt.plot(xall, yall, 'k-', label='Truth function')
    inter_par = interpolation.Inter_par(method="NPS")
    ##### Plot the discrete search function  #####
    [interpo_par, yp] = interpolation.regressionparametarization(xE, yE, SigmaT, inter_par)
    sd = np.amin((yp, 2 * yE - yp), 0) - L * SigmaT
    plt.scatter(xE[0], sd, color='r', marker='s', s=15, label='Discrete search function')
    ##### Plot the continuous search function  #####
    xi = np.hstack([xE, xU])
    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)
    Sc = np.array([])
    
    K0 = np.ptp(yE, axis=0) 
    for ii in range(len(tri)):
        temp_x = xi[:, tri[ii, :]]
        x = np.arange(temp_x[0, 0], temp_x[0, 1]+0.005, 0.005)
        temp_Sc = np.zeros(len(x))
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x)):
            temp_Sc[jj] = interpolation.interpolate_val(x[jj], inter_par) - K * K0 * (R2 - np.linalg.norm(x[jj] - xc) ** 2)
        Sc = np.hstack([Sc, temp_Sc])          
    x_sc = np.linspace(0, np.max(xi), len(Sc))
    plt.plot(x_sc, Sc, 'g--', label='Continuous search function')
    #####  Plot the separable point for continuous search function  #####
    sscc = np.zeros(xE.shape[1])
    for j in range(len(sscc)):
        for i in range(len(x_sc)):
            if np.linalg.norm(x_sc[i] - xE[0, j]) < 6*1e-3:
                sscc[j] = Sc[i]
    plt.scatter(xE[0], sscc, color='green', s=10)
    #####  Plot the errorbar  #####
    plt.errorbar(xE[0], yE, yerr=SigmaT, fmt='o', label='Function evaluation')
    ########    plot the regression function   ########
    yrall = np.arange(-0.05, 1.01, 0.001)
    for i in range(len(yrall)):
        yrall[i] = interpolation.interpolate_val(xall[i], inter_par)
    plt.plot(xall, yrall, 'b--', label='Regression function')
    
    if Utils.mindis(xc_min, xE)[0] < 1e-6:
        plt.annotate('Mesh Refinement Iteration', xy=(0.6, 3.5), fontsize=8)
    else:
        if yc < yd:
            #####  Plot the minimum of continuous search function as star  #####
            plt.scatter(xc_min, np.min(Sc), marker=(5, 2))
            #####  Plot the arrowhead  point at minimum of continuous search funtion  #####
            plt.annotate('Identifying sampling', xy=(xc_min, np.min(Sc)), 
                         xytext=(np.abs(xc_min-0.5), np.min(Sc)-1),
                                arrowprops=dict(facecolor='black', shrink=0.05))
                        
        else:
            #####  Plot the arrowhead  point at minimum of discrete search funtion  #####
            xd = xE[:, np.argmin(sd)]
            plt.annotate('Supplemental Sampling', xy=(xd[0], np.min(sd)), xytext=(np.abs(xd[0]-0.5), np.min(sd)-1),
                         arrowprops=dict(facecolor='black', shrink=0.05))
    ##### Plot the information about parameters of each iteration  #####
    plt.annotate('#Iterations = ' + str(int(num_iter)), xy=(0, 3.5), fontsize=8)
    plt.annotate('Mesh size = ' + str(int(Nm)), xy=(0.35, 3.5), fontsize=8, color='b')
    plt.annotate('K = ' + str(int(K)), xy=(0.35, 3), fontsize=8, color='b')
    plt.annotate('Range(Y) = ' + str(float("{0:.4f}".format(K0))), xy=(0.28, 2.5), fontsize=8, color='b')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", prop={'size': 6}, borderaxespad=0.)
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.grid()
    ##### Check if the folder 'plot' exists or not  #####
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/cd_movie"   # -5 comes from the length of '/dogs'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)  
    # Save fig
    plt.savefig(plot_folder + '/cd_movie' + str(num_iter) +'.png', format='png', dpi=1000)    
    return 


def plot_delta_dogs(xE, Nm, plot_class, Alg, Kini=0):
    '''
      This function is set for plotting Delta-DOGS algorithm without dimension reduction on toy problem, 
    e.g. Schwefel function or Schwefel + Quadratic.
      Also this function can plot continuous search function for cosntantK or Adaptive K. Parameters containing:
    continuous search function, discrete search function, regression function, truth function and function evaluation.
    
    :param xE           :  The data of original input, evaluated points.
    :param Kini         :  The initial tuning parameter for ConstantK search function.
    :param Nm           :  Current mesh size.
    :param plot_class   :  The information for plotting.
    :param Alg          :
    :return: Plot for each iteration of Delta-DOGS algorithm, save the fig at 'plot/DDOGS' folder.
    '''
    fun_arg = plot_class.fun_arg
    num_ini = plot_class.num_ini
    sc = plot_class.type
    ff = plot_class.ff
    init_comp = plot_class.init_comp
    store_plot = plot_class.store_plot

    # Plot the truth function
    n = xE.shape[0]
    fig = plt.figure()
    x = y = np.linspace(-0.05, 1.05, 500)
    X, Y = np.meshgrid(x, y)

    Z, y0, xmin = function_eval(X, Y, fun_arg)
    # Make the contour plot for the original function.    
    l = np.linspace(np.min(Z), np.max(Z), 30)
    plt.contourf(X, Y, Z, cmap='gray', levels=l)
    plt.colorbar()
    
    # Plot the initial points.
    plt.scatter(xE[0, :num_ini], xE[1, :num_ini], c='w', label='Initial points', s=10)
    
    # Plot the rest point.
    plt.scatter(xE[0, num_ini:], xE[1, num_ini:], c='b', label='2D search points', s=10)
    
    # Number of iteration:
    num_iter = int(xE.shape[1] - num_ini + np.log(Nm/8) / np.log(2))
    # Define the titile for the plot.
    if sc == "ConstantK":
        plt.title(r"$\Delta$-DOGS($\Lambda$) " + sc + ': K = ' + str(Kini) + ' ' + str(num_iter) + "th Iteration: " + 'MeshSize = ' + str(Nm), y=1.05)
    elif sc == "AdaptiveK":
        plt.title(r"$\Delta$-DOGS($\Lambda$): " + sc + ' ' + str(num_iter) + "th Iteration: " + 'MeshSize = ' + str(Nm), y=1.05)

    # Plot the latest point.
    plt.scatter(xE[0, -1], xE[1, -1], c='r', label='Current Evaluated point', s=10)
    search_boundary_plot(plot_class.xU_plot)
    # Plot the reduced regression model
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    # Plot the global minimizer of function.
    plt.scatter(xmin[0], xmin[1], c='y', marker='*', s=3, label='Global minimum')
    plt.grid()
    # Make the labels of the plot.
    plt.legend(bbox_to_anchor=(0., 1.06, 0., 0), loc=2,
               ncol=4, prop={'size': 6}, shadow=True)
    ##### Check if the folder 'plot/DDOGS/ff' exists or not  #####
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/" + Alg + str(ff)      # -5 comes from the length of '/dogs'

    if init_comp == 0:
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_class.init_comp = 1
    # Save fig
    if store_plot == 1:
        plt.savefig(plot_folder + '/pic' + str(num_iter) +'.png', format='png', dpi=250)
    plt.close(fig)
    return plot_class


def search_boundary_plot(xU):
    next = xU[:, 0].reshape(-1, 1)
    xU_pure = np.delete(xU, 0, 1)
    for i in range(xU.shape[1]-1):
        nn_index = np.argmin(np.linalg.norm(xU_pure - next, axis=0))
        plot_current = np.hstack(( next, xU_pure[:, nn_index].reshape(-1, 1) ))
        plt.plot(plot_current[0, :], plot_current[1, :], c='b')
        next = xU_pure[:, nn_index].reshape(-1, 1)
        xU_pure = np.delete(xU_pure, nn_index, 1)
    plot_current = np.hstack(( next, xU[:, 0].reshape(-1, 1) ))
    plt.plot(plot_current[0, :], plot_current[1, :], c='b')
    return


def plot_delta_dogs_dr(xE, xc, Nm, plot_class, xr, w, Kini=0):
    '''
    This function is set for plotting Delta-DOGS algorithm or Delta-DOGS with DR on toy problem, e.g. Schwefel function
    or Schwefel + Quadratic.
    Also this function can plot continuous search function for cosntantK and Adaptive K. Parameters containing:
    continuous search function, discrete search function, regression function, truth function and function evaluation.
    
    :param xE: The data of original input, evaluated points.
    :param yE: Function evaluation of xE.
    :param num_iter: Represents the number of initial evaluated points.
    :param Nm: Current mesh size.
    :param Nm_p1: Represents the number of mesh refinement at 1D reduced model.
    :param plot_class: Contain information of plot
    :param Kini: The initial tuning parameter for ConstantK search function. Right now DR is only fixed for AdaptiveK.
    :return: Plot for each iteration of Delta-DOGS algorithm, save the fig at 'plot/DimRec' folder.
    '''
    # Plot the truth function
    fun_arg = plot_class.fun_arg
    num_ini = plot_class.num_ini
    sc = plot_class.type
    ff = plot_class.ff
    init_comp = plot_class.init_comp
    store_plot = plot_class.store_plot

    n = xE.shape[0]
    fig = plt.figure()
    x = y = np.linspace(-0.05, 1.05, 500)
    X, Y = np.meshgrid(x, y)
    Z, y0, xmin = function_eval(X, Y, fun_arg)
    l = np.linspace(np.min(Z), np.max(Z), 30)
    plt.contourf(X, Y, Z, cmap='gray', levels=l)
    
    plt.colorbar()
    # Plot the initial points.
    plt.scatter(xE[0, :num_ini], xE[1, :num_ini], c='w', label='Initial points', s=6)
    num_iter = int(xE.shape[1] - num_ini + np.log(Nm/8) / np.log(2))
    # if sc == "ConstantK":
    #     plt.title(r"DR $\Delta$-DOGS($\Lambda$): " + sc + ': K = ' + str(Kini) + ' ' + str(num_iter) + "th Iteration: MS = " + str(Nm), y=1.05)
    # elif sc == "AdaptiveK":
    #     plt.title(r"DR $ \Delta$-DOGS($\Lambda$): " + sc + ' ' + str(num_iter) + "th Iteration: MS = " + str(Nm), y=1.05)

    # Plot the search points
    plt.scatter(xE[0, num_ini:], xE[1, num_ini:], c='g', label='1D Reduced', s=6)
    # Plot the latest point.
    plt.scatter(xc[0, 0], xc[1, 0], c='r', label='Current Evaluate point', s=6)
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    # Plot the global minimum
    plt.scatter(xmin[0], xmin[1], c='y', marker='*', label='Global minimum', s=2)

    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.06, 0., 0), loc=2,
               ncol=4, prop={'size': 6}, shadow=True)
    ##### Check if the folder 'plot' exists or not  #####
    name = 'DR/'
    plot_folder = folder_path(name, ff) + '/DimRed'
    if init_comp == 0:
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_class.init_comp = 1
    # Save fig
    if store_plot == 1:
        plt.savefig(plot_folder + '/DR' + str(num_iter) +'.png', format='png', dpi=250)
    #####################################  Illustration  ##############################
    illus_init_comp = plot_class.illustration_init_comp
    if illus_init_comp < 0:  # No illustration plot
        plt.close(fig)
    else:
        plot_folder = folder_path(name, ff) + "/illustration"
        if illus_init_comp == 0:  # Initial the illustration plot
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            plot_class.illustration_init_comp = 1
        #################################  Show illustration of DR  #################################
        # plot the W_1^T*X_E=0.
        x_linewT = np.linspace(0, 1, 100)
        y_linewT = - w[0] * x_linewT / w[1]
        plt.plot(x_linewT, y_linewT, color="orange", label=r'$X_R$=0')
        # plot the new coordinate that does not exist in 2D parameter space
        x_line = np.linspace(-0.05, 1.05, 100)
        y_line = w[1] / w[0] * (x_line - 0.5) + 0.5  # ?? w[0]/w[1] or reverse?
        plt.plot(x_line, y_line, c='k', label='New coordinate')

        # Plot the projected points on reduced subspace, passing [0.5, 0.5]
        for ii in range(xE.shape[1]):
            xii = xE[:, ii].reshape(-1, 1)
            # orth_xii is the intersection of two lines.
            orth_xii = (w[1] / w[0] * 0.5 + w[0] / w[1] * xii[0] + xii[1] - 0.5) * (w[0] * w[1]) / (
                    w[0] ** 2 + w[1] ** 2)
            xii_line = np.linspace(xii[0, 0], orth_xii[0], 10)
            yii_line = -w[0] / w[1] * xii_line + w[0] / w[1] * xii[0, 0] + xii[1, 0]
            plt.plot(xii_line, yii_line, '--', c='b', label='Projection', linewidth=1.5)
        # Full model search constraint
        xline = np.linspace(0, 1, 100)
        b = xr[0, 0]
        yline = (b - w[0] * xline) / w[1]
        plt.plot(xline, yline, c='m', linewidth=1.5, label=r'Full search constraint $X_R=x_r$')
        #################################################################################
        plt.legend(bbox_to_anchor=(0., 1.06, 0., 0), loc=2,
                   ncol=4, prop={'size': 6}, shadow=True)
        if store_plot == 1:
            plt.savefig(plot_folder + '/DRillustration' + str(num_iter) + '.png', format='png', dpi=250)
        plt.close(fig)
    return plot_class


def evaluate_interpolation_surface(inter_par):
    fig = plt.figure()
    x = y = np.linspace(-0.05, 1.05, 250)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            x_eval = np.array([ X[i, j], Y[i, j] ])
            Z[i, j] = interpolation.interpolate_val(x_eval, inter_par)
    l = np.linspace(np.min(Z), np.max(Z), 30)
    plt.title('Interpolation surface')
    plt.contourf(X, Y, Z, cmap='gray', levels=l)
    plt.grid()
    plt.colorbar()
    return


def triplot_2D(xE):
    plt.figure()
    plt.grid()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    xE_plot = np.hstack((xE, xE[:,0].reshape(-1, 1)))
    plt.plot(xE_plot[0, :], xE_plot[1, :], c='b')
    plt.show()


def reduced_interpolation(inter_par_asm, reduced_interval, plot_class, num_iter, xmin, w, y0, xR, yp_r):
    ff = plot_class.ff
    x = np.linspace(np.min(reduced_interval), np.max(reduced_interval), 500)
    y = np.zeros(x.shape)
    for i in range(y.shape[0]):
        y[i] = interpolation.interpolate_val(x[i], inter_par_asm)
    fig = plt.figure()
    plt.grid()
    plt.plot(x, y, label='Interpolation')
    xrmin = np.dot(w.T, xmin.reshape(-1, 1))
    # plot the global minimum's 1D projection vertical line.
    plt.axvline(x=xrmin[0, 0], color='g', linestyle='--', label='2D Global min projection')
    # Indicate the minimum of the interpolation.
    plt.scatter(x[np.argmin(y)], np.min(y), marker=(5, 2), color='yellow')

    # The scatter plot to define the interpolation.
    plt.scatter(xR[0], yp_r, c='r', s=6)

    # plot the horizontal line of minimization target y0.
    plt.axhline(y=y0, color='r', linestyle='--')

    plt.title("Interpolation on 1D reduced model of ASM")
    plt.legend()
    name = 'DR/'
    plot_folder = folder_path(name, ff) + "/Interpolation"

    if plot_class.itrp_init_comp == 0:
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_class.itrp_init_comp = 1
    # Save fig
    plt.savefig(plot_folder + '/Itrp1D' + str(num_iter) +'.png', format='png', dpi=250)
    plt.close(fig)
    return plot_class


# generate sc, p, f for regular DeltaDOGS scheme
def sc_intrpln_1D_simplex_plot(inter_par_asm, xi, yE, fun_arg, type):
    # xi should be xi[:, tri[ind, :]]
    # Just for one 1D simplex, plot interpolation + sc.
    # If the DeltaDOGS optm in reduced space is not good, this is for showing what's wrong
    n = inter_par_asm.xi.shape[0]
    K0 = np.ptp(yE, axis=0)
    fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
    func_eval = partial(Utils.fun_eval, fun, lb, ub)
    y0 -= 0.015
    fig, ax1 = plt.subplots()
    ##### Plot the continuous search function  #####
    if type == 'full':
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
        Sc = np.array([])
        invSc = np.array([])
        K0 = np.ptp(yE, axis=0)
        x_min = np.copy(xi[:, 0])
        sc_min = 1e+10
        for ii in range(len(tri)):
            temp_x = np.copy(xi[:, tri[ii, :]])
            x = np.arange(temp_x[0, 0]+0.0001, temp_x[0, 1]-0.0001, 0.0001)
            temp_Sc = np.zeros(len(x))
            temp_invSc = np.zeros(len(x))
            R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
            for jj in range(len(x)):
                p = interpolation.interpolate_val(x[jj], inter_par_asm) - y0
                e = (K0 * (R2 - np.linalg.norm(x[jj] - xc) ** 2))
                temp_Sc[jj] = p / e
                temp_invSc[jj] = - e / p
            Sc = np.hstack([Sc, temp_Sc])
            invSc = np.hstack((invSc, temp_invSc))
            if temp_Sc.min() < sc_min:
                sc_min = temp_Sc.min()
                xmin = np.copy(x[np.argmin(temp_Sc)])
            ax1.plot(x, temp_Sc, 'g--', label=r'Continuous search function $s_c(x)$')

        x_sc = np.linspace(0, np.max(xi), len(Sc))
        x = np.copy(x_sc)
        temp_Sc = np.copy(Sc)
        temp_invSc = np.copy(invSc)

    else:
        x = np.linspace(xi.min()+0.0001, xi.max()-0.0001, 1000)
        temp_Sc = np.zeros(len(x))
        temp_invSc = np.zeros(len(x))
        R2, xc = Utils.circhyp(xi, n)
        for jj in range(len(x)):
            p = interpolation.interpolate_val(x[jj], inter_par_asm) - y0
            e = (K0 * (R2 - np.linalg.norm(x[jj] - xc) ** 2))
            temp_Sc[jj] = p / e
            temp_invSc[jj] = - e / p

    xRplot = np.copy(x)
    p_sscc = np.copy(temp_Sc)
    sc_min = np.min(p_sscc)
    x_min = xRplot[np.argmin(p_sscc)]

    # ax1.plot(xRplot, p_sscc, 'g--', label=r'Continuous search function $s_c(x)$')
    ax1.scatter(x_min, sc_min, c='r')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax1.set_ylabel(r'$s_c(x)$', color='g')
    ax1.tick_params('y', colors='g')
    ########    plot the interpolation function   ########
    ax2 = ax1.twinx()
    yrall = np.zeros(xRplot.shape)
    y_true = np.zeros(xRplot.shape)

    for i in range(len(xRplot)):
        yrall[i] = interpolation.interpolate_val(xRplot[i], inter_par_asm)
        y_true[i] = func_eval(xRplot[i])
    ax2.plot(xRplot, yrall, 'b--', label='Interpolation function')
    ax2.plot(xRplot, y_true, 'k', label='True function')
    ax2.set_ylabel('p(x)', color='b')
    ax2.scatter(xi[0], yE, c='k')
    ax2.tick_params('y', colors='b')

    print('min(sc) = ' + str(float("{0:.4f}".format(np.min(abs(p_sscc))))))
    print('argmin(sc) = ' + str(float("{0:.4f}".format(xRplot[np.argmin(abs(p_sscc))]))))
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
    fig.tight_layout()
    plt.savefig('zz.png', format='png', dpi=250)

    ########################################### - 1 / sc plot #############################
    # fig2, ax3 = plt.subplots()
    # xRplot2 = scipy.delete(x, dlt1, 0)
    # invp_sc = scipy.delete(temp_invSc, dlt1, 0)
    # ax3.plot(xRplot2, invp_sc, 'g--', label=r'$-\frac{1}{s_c(x)}$')
    # ax3.set_ylabel(r'$-\frac{1}{s_c(x)}$', color='g')
    # ax3.tick_params('y', colors='g')
    # plt.legend()
    #
    # ax4 = ax3.twinx()
    # ax4.plot(xRplot, yrall, 'b--', label='Interpolation')
    # ax4.set_ylabel('p(x)', color='b')
    # ax4.tick_params('y', colors='b')
    #
    # print('min oppinv (sc) = ' + str(float("{0:.4f}".format(np.min(invp_sc)))))
    # print('argmin oppinv (sc) = ' + str(float("{0:.4f}".format(xRplot2[np.argmin(invp_sc)]))))
    # plt.grid()
    # plt.legend()
    # fig.tight_layout()
    plt.show()
    return


def subplot_delta_dogs_dr(xE, xc_pack, Nm, plot_class, xr, w, inter_par_asm, reduced_interval, xgmin, epsilon, x0):
    fun_arg = plot_class.fun_arg
    num_ini = plot_class.num_ini
    sc = plot_class.type
    subplot_init_comp = plot_class.subplot_init_comp
    store_plot = plot_class.store_plot

    # xc_pack
    xc = xc_pack['xc']
    xc_grid = xc_pack['xc_grid']
    xc_eval = xc_pack['xc_eval']

    # plt.subplot(2, 2, 1)
    plt.subplot(121, aspect='equal', adjustable='box-forced')
    x = y = np.linspace(-0.05, 1.05, 500)
    X, Y = np.meshgrid(x, y)
    Z, y0, xmin = function_eval(X, Y, fun_arg)
    l = np.linspace(np.min(Z), np.max(Z), 30)
    plt.contourf(X, Y, Z, cmap='gray', levels=l)

    # Plot the initial points.
    plt.scatter(xE[0, :num_ini], xE[1, :num_ini], c='w', s=4)#, label='Initial points')
    num_iter = int(xE.shape[1] - num_ini + np.log(Nm/8) / np.log(2))
    plt.title(str(num_iter) + "th Iteration: MS = " + str(Nm), y=1.05)

    plt.scatter(xE[0, num_ini:], xE[1, num_ini:], c='g', label='1D Reduced', s=6)
    # Plot the latest point.
    plt.scatter(xc_eval[0, 0], xc_eval[1, 0], c='r', s=4)#, label='Current Evaluate point')
    # Plot the DR+DeltaDOGS search grid points, may evaluate support points
    plt.scatter(xc_grid[0], xc_grid[1], c='orange', s=4, label='Grid')
    # Plot the search points:
    plt.scatter(xc[0], xc[1], c='c', s=4, label=r'$X_c$')

    # Plot the intial point for minimization
    plt.scatter(x0[0], x0[1], c='yellow', s=4, label='FirstGuess')

    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    # Plot the global minimum
    plt.scatter(xmin[0], xmin[1], c='y', marker='*', s=3)#, label='Global minimum')

    # Full model search constraint
    xline = np.linspace(0, 1, 500)
    b = xr[0, 0]
    yline = (b - w[0] * xline) / w[1]
    yuline = ((b+epsilon)-w[0] * xline) / w[1]
    ylline = ((b-epsilon)-w[0] * xline) / w[1]
    delete = []
    dell = []
    delu = []
    for i in range(yline.shape[0]):
        if yline[i] < -1e-3 or yline[i] > 1+1e-3:
            delete.append(i)
        if yuline[i] < -1e-3 or yuline[i] > 1+1e-3:
            delu.append(i)
        if ylline[i] < -1e-3 or ylline[i] > 1+1e-3:
            dell.append(i)
    # upper bound of discrete search
    xuline = scipy.delete(xline, delu, 0)
    yuline = scipy.delete(yuline, delu, 0)
    # lower bound of discrete search
    xlline = scipy.delete(xline, dell, 0)
    ylline = scipy.delete(ylline, dell, 0)
    # the exact line constraint of discrete search
    xline = scipy.delete(xline, delete, 0)
    yline = scipy.delete(yline, delete, 0)
    plt.plot(xline, yline, c='b', linewidth=0.8) #, label=r'Full search constraint $X_R=x_r$')
    plt.plot(xlline, ylline, 'b--', linewidth=0.8)
    plt.plot(xuline, yuline, 'b--', linewidth=0.8)

    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.06, 0., 0), loc=2,
               ncol=4, prop={'size': 6}, shadow=True)
    ############################################################################################
    # plt.subplot(2, 2, 2)
    plt.subplot(122, adjustable='box-forced')
    ff = plot_class.ff
    ##### Plot the continuous search function  #####
    xi = np.copy(reduced_interval)
    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)
    Sc = np.array([])
    n = xi.shape[0]
    for ii in range(len(tri)):
        temp_x = xi[:, tri[ii, :]]
        x = np.arange(temp_x[0, 0], temp_x[0, 1] + 0.005, 0.005)
        temp_Sc = np.zeros(len(x))
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x)):
            dis = (R2 - np.linalg.norm(x[jj] - xc) ** 2)
            if dis < 1e-6:
                dis = 1e-6
            temp_Sc[jj] = (interpolation.interpolate_val(x[jj], inter_par_asm) - y0) / dis
        Sc = np.hstack([Sc, temp_Sc])
    mu = np.mean(Sc)
    std = np.std(Sc)
    dlt = []
    for i in range(Sc.shape[0]):
        if np.abs(Sc[i] - mu) > std:
            dlt.append(i)
    Sc = scipy.delete(Sc, dlt, 0)
    x_sc = np.linspace(0, np.max(xi), len(Sc))
    #####  Plot the separable point for continuous search function  #####
    sscc = np.zeros(xi.shape[1])
    for j in range(len(sscc)):
        for i in range(len(x_sc)):
            if np.linalg.norm(x_sc[i] - xi[0, j]) < 6 * 1e-3:
                sscc[j] = Sc[i]


    x = np.linspace(np.min(reduced_interval), np.max(reduced_interval), 500)
    y = np.zeros(x.shape)
    for i in range(y.shape[0]):
        y[i] = interpolation.interpolate_val(x[i], inter_par_asm)
    plt.grid()
    plt.plot(x, y, label='Interpolation')

    # Scaling continuous function for plotting in the same figure.
    factor = np.mean(np.abs(Sc))/np.mean(np.abs(y))
    Sc = np.divide(Sc, factor)
    plt.plot(x_sc, Sc, 'g--', label='Continuous search function')
    sscc = np.divide(sscc, factor)
    plt.scatter(xi[0], sscc, color='green', s=10)
    scmin = np.min(Sc)
    xscmin = x_sc[np.argmin(Sc)]
    scmin = np.divide(scmin, factor)
    plt.scatter(xscmin, scmin, marker='o', label='Min(Sc)')

    # xgmin - the global minimum of original full space function.
    xrmin = np.dot(w.T, xgmin.reshape(-1, 1))
    # plot the global minimum's 1D projection place.
    plt.axvline(x=xrmin[0, 0], color='k', linestyle='--', label='2D Global min projection')
    # plot the minimum horizontal line.
    plt.axhline(y=y0, color='r', linestyle='--')

    plt.title("Interpolation on 1D reduced model of ASM")
    plt.legend(prop={'size': 5})

    name = 'DR/'
    plot_folder = folder_path(name, ff) + "/subplot"
    if subplot_init_comp == 0:
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_class.subplot_init_comp = 1
    # Save fig
    if store_plot == 1:
        plt.savefig(plot_folder + '/DR' + str(num_iter) +'.png', format='png', dpi=250)
    plt.close()
    return plot_class


def dr_plot_2D_trikind(plot_class, xE, xc_pack, Nm, xr, w, inter_par_asm, xR,
                                                             num_iter, fun_arg, xr_intrpn, yp_r, epsilon, x0):
    xc_eval = xc_pack['xc_eval']
    x = y = np.linspace(-0.05, 1.05, 500)
    X, Y = np.meshgrid(x, y)
    Z, y0, xmin = function_eval(X, Y, fun_arg)
    if plot_class.init_comp >= 0:
        plot_class = plot_delta_dogs_dr(xE, xc_eval, Nm, plot_class, xr, w)
    if plot_class.itrp_init_comp >= 0:
        plot_class = reduced_interpolation(inter_par_asm, xR, plot_class, num_iter, xmin, w, y0, xr_intrpn, yp_r)
    if plot_class.subplot_init_comp >= 0:
        plot_class = subplot_delta_dogs_dr(xE, xc_pack, Nm, plot_class, xr, w, inter_par_asm, xR, xmin, epsilon, x0)
    return plot_class


def discrete_eval(inter_par):
    xE = inter_par.xi
    x = np.linspace(0, 1, 500)
    y = np.linspace(0, 1, 500)
    X, Y = np.meshgrid(x, y)
    n, m = X.shape
    Z = np.zeros(X.shape)
    for i in range(n):
        for j in range(m):
            temp = np.array([[X[i, j]], [Y[i, j]]])
            p = interpolation.interpolate_val(temp, inter_par)
            e = DR_adaptiveK.min_MDD(x, xE)
            if e < 1e-6:
                e = 1e-6
            Z[i, j] = p / e
    plt.figure()
    plt.contourf(X, Y, Z, cmap='gray')
    plt.colorbar()
    plt.scatter(xE[0, :], xE[1, :], c='r', label='initial')
    plt.show()
    return


def delaunay_uncertainty_1Ddomain(xi, yE):
    '''
    Calculate the uncertainty function in the entire domain.
    :param xi:             Evaluated points set S.
    :param uncertainty:
    :return: a vector e represents the uncertainty value on each point of x.
    '''
    n = xi.shape[0]
    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)
    K0 = np.ptp(yE, axis=0)
    for ii in range(len(tri)):
        temp_x = np.copy(xi[:, tri[ii, :]])
        x = np.arange(temp_x[0, 0] + 0.0001, temp_x[0, 1] - 0.0001, 0.001)
        e_simplex = np.zeros(len(x))
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x)):
            e_simplex[jj] = K0 * (R2 - np.linalg.norm(x[jj] - xc) ** 2)
        if ii == 0:
            e = np.copy(e_simplex)
        else:
            e = np.hstack((e, e_simplex))
    return e


def sc_interp_1D_separate_delaunay_adaptivek(inter_par, xi, yE, fun_arg):
    n = inter_par.xi.shape[0]
    fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
    func_eval = partial(Utils.fun_eval, fun, lb, ub)

    plt.figure()
    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)

    num_plot_points = 2000
    xe_plot = np.zeros((tri.shape[0], num_plot_points))
    e_plot  = np.zeros((tri.shape[0], num_plot_points))
    sc_plot = np.zeros((tri.shape[0], num_plot_points))

    for ii in range(len(tri)):
        temp_x = np.copy(xi[:, tri[ii, :]])
        x_ = np.linspace(temp_x[0, 0], temp_x[0, 1], num_plot_points)
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x_)):
            p = interpolation.interpolate_val(x_[jj], inter_par) - y0
            e_plot[ii, jj] = (R2 - np.linalg.norm(x_[jj] - xc) ** 2)
            sc_plot[ii, jj] = p / e_plot[ii, jj]
    sc_min = np.min(sc_plot, axis=1)
    index_r = np.argmin(sc_min)
    index_c = np.argmin(sc_plot[index_r, :])
    sc_min_x = xe_plot[index_r, index_c]
    sc_min = min(np.min(sc_plot, axis=1))

    # Plotting uncertainty function, delete elements when sc(x) exceeds the range

    for ii in range(len(tri)):
        dlt = []
        temp_x = np.copy(xi[:, tri[ii, :]])
        x_ = np.linspace(temp_x[0, 0], temp_x[0, 1], num_plot_points)

        for i in range(sc_plot.shape[1]):
            if sc_plot[ii, i] > 200:
                dlt.append(i)
        x_ = scipy.delete(x_, dlt, 0)
        temp_Sc = scipy.delete(sc_plot[ii, :], dlt, 0)
        plt.plot(x_, temp_Sc, 'g--', label=r'Continuous search function $s_c(x)$')

    plt.scatter(sc_min_x, sc_min, c='r')
    plt.tick_params('y', colors='g')
    plt.savefig('Iter' + str(yE.shape[0]) + 'sc.eps', format='eps', dpi=1000)
    plt.close()
    ############ uncertainty ############
    plt.figure()
    for i in range(len(tri)):
        plt.plot(xe_plot[i, :], e_plot[i, :], 'g')
    plt.tick_params('y', colors='b')
    plt.ylim(0, 0.5)
    plt.savefig('Iter' + str(yE.shape[0]) + 'uncertainty.eps', format='eps', dpi=1000)

    ############ interpolation ############
    plt.figure()
    xRplot = np.linspace(0, 1, 1000)
    yrall = np.zeros(xRplot.shape)
    y_true = np.zeros(xRplot.shape)

    for i in range(len(xRplot)):
        yrall[i] = interpolation.interpolate_val(xRplot[i], inter_par)
        y_true[i] = func_eval(xRplot[i])
    plt.plot(xRplot, yrall, 'b--', label='Interpolation function')
    plt.plot(xRplot, y_true, 'k', label='True function')
    plt.scatter(xi[0], yE, c='k')
    plt.tick_params('y', colors='b')
    plt.savefig('Iter' + str(yE.shape[0]) + 'f_p.eps', format='eps', dpi=1000)
    return


def sc_interp_1D_separate_delaunay_constantk(inter_par, xi, yE, K, fun_arg):

    n = inter_par.xi.shape[0]
    fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
    func_eval = partial(Utils.fun_eval, fun, lb, ub)

    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)

    num_plot_points = 2000
    xe_plot = np.zeros((tri.shape[0], num_plot_points))
    e_plot  = np.zeros((tri.shape[0], num_plot_points))
    sc_plot = np.zeros((tri.shape[0], num_plot_points))

    for ii in range(tri.shape[0]):
        temp_x = np.copy(xi[:, tri[ii, :]])

        x_ = np.linspace(temp_x[0, 0], temp_x[0, 1], num_plot_points)
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x_)):
            p = interpolation.interpolate_val(x_[jj], inter_par)

            e_plot[ii, jj] = (R2 - np.linalg.norm(x_[jj] - xc) ** 2)
            sc_plot[ii, jj] = p - K * e_plot[ii, jj]

        xe_plot[ii, :] = x_

    sc_min = np.min(sc_plot, axis=1)
    index_r = np.argmin(sc_min)
    index_c = np.argmin(sc_plot[index_r, :])
    sc_min_x = xe_plot[index_r, index_c]
    sc_min = min(np.min(sc_plot, axis=1))

    # plot the uncertainty
    fig = plt.figure()
    if xi.shape[1] < 5:
        amp = 1
    else:
        amp = 10
    for i in range(len(tri)):
        plt.plot(xe_plot[i, :], amp * e_plot[i, :], 'g')
    plt.tick_params('y', colors='b')
    plt.ylim(0, 0.5)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig('Iter' + str(yE.shape[0]) + 'uncertainty.png', format='png', dpi=1000)
    plt.close(fig)

    ############ interpolation ############
    fig = plt.figure()
    xRplot = np.linspace(0, 1, 1000)
    yrall = np.zeros(xRplot.shape)
    y_true = np.zeros(xRplot.shape)

    for i in range(len(xRplot)):
        yrall[i] = interpolation.interpolate_val(xRplot[i], inter_par)
        y_true[i] = func_eval(xRplot[i])
    plt.plot(xRplot, yrall, 'b--', label='Interpolation function')
    plt.plot(xRplot, y_true, 'k', label='True function')
    # plot the s(x)
    for i in range(len(tri)):
        plt.plot(xe_plot[i, :], sc_plot[i, :], c='r')
    # plot the evaluated data points
    plt.scatter(xi[0], yE, c='k')
    plt.gca().axes.get_yaxis().set_visible(False)
    # plot the minimizer of s(x)
    plt.scatter(sc_min_x, sc_min, c='r')
    plt.tick_params('y', colors='g')
    plt.ylim(-2, 2)
    plt.savefig('Iter' + str(yE.shape[0]) + 'sc.png', format='png', dpi=1000)
    return


def sd_p_f_e_1D_seperate(inter_par, yE, fun_arg, Dtype, alg_name, ff):
    '''
    Generate results for 1D DeltaDOGS.
    :param inter_par:
    :param yE:
    :param fun_arg:
    :param Dtype:
    :param alg_name:
    :param ff:
    :return:
    '''
    n = inter_par.xi.shape[0]
    xE = np.copy(inter_par.xi)
    fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
    func_eval = partial(Utils.fun_eval, fun, lb, ub)

    plot_folder = folder_path(alg_name, ff)
    ####################  Interpolation  & Truth Function  ####################
    fig1 = plt.figure()
    dx = 0.0001
    x = np.arange(0 + dx, 1 - dx, dx)
    yr = np.zeros(x.shape)
    y_true = np.zeros(x.shape)
    for i in range(x.shape[0]):
        yr[i] = interpolation.interpolate_val(x[i], inter_par)
        y_true[i] = func_eval(x[i])
    plt.plot(x, yr, 'b--', label='Interpolation function')
    plt.plot(x, y_true, 'k', label='True function')
    plt.scatter(xE[0], yE, c='k')
    plt.tick_params('y', colors='b')
    plt.grid()
    plt.title(alg_name[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fname +
                ': Interpolation and truth function')
    plt.savefig(plot_folder + '/Iter' + str(yE.shape[0]) + '_f&p.eps', format='eps', dpi=1000)
    plt.close(fig1)
    ####################  Uncertainty  Function  ####################
    fig2 = plt.figure()
    if Dtype['name'] == 'square_root_2norm':
        c = Dtype['c']
        uncertainty = partial(discrete_min.square_root_L2distance, xE, c)
    e = np.zeros(x.shape)
    for i in range(x.shape[0]):
        e[i] = max(1e-10, uncertainty(x[i]))

    plt.plot(x, e, 'g')
    plt.tick_params('y', colors='b')
    plt.ylim([0, 1])
    plt.grid()
    plt.title(alg_name[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fname +
                ': Uncertainty function')
    plt.savefig(plot_folder + '/Iter' + str(yE.shape[0]) + '_uncertainty.eps', format='eps', dpi=1000)
    plt.close(fig2)
    ####################  Discrete Search Function  ####################
    sd = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        sd[i] = (yr[i] - y0) / e[i]

    sd_min = np.min(sd)
    xmin = x[np.argmin(sd)]
    fig3 = plt.figure()
    dlt = []
    for i in range(sd.shape[0]):
        if sd[i] > 200:
            dlt.append(i)
    x = scipy.delete(x, dlt, 0)
    sd = scipy.delete(sd, dlt, 0)
    plt.plot(x, sd, 'g--', label=r'Discrete search function $s_d(x)$')
    plt.scatter(xmin, sd_min, c='r')
    plt.tick_params('y', colors='g')
    plt.grid()
    plt.title(alg_name[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fname +
                ': Discrete search function')
    plt.savefig(plot_folder + '/Iter' + str(yE.shape[0]) + '_sc.eps', format='eps', dpi=1000)
    plt.close(fig3)
    return


def square_root_L2distance(xE, x, c, s):
    x = x.reshape(-1, 1)
    nn = xE[:, np.argmin(np.linalg.norm(xE - x, axis=0))].reshape(-1, 1)
    dis = (np.linalg.norm(nn - x)**2 + c) ** s - c ** s
    return dis


def square_root_L2norm_1Dplot(xE, c, s):
    x = np.linspace(0, 1, 1000)
    e = np.zeros(x.shape)
    for i in range(x.shape[0]):
        e[i] = square_root_L2distance(xE, x[i], c, s)
    fig = plt.figure()
    plt.plot(x, e, 'g')
    plt.title(r'$e(x)=(||x-NN(x)||_2^2+c)^s - c^s$, ' + 's = ' + str(s) + ', c = ' + str(c))
    plt.show()
    plt.close(fig)
    return


def exponential_distance(xE, x):
    x = x.reshape(-1, 1)
    nn = xE[:, np.argmin(np.linalg.norm(xE - x, axis=0))].reshape(-1, 1)
    dis = np.exp(np.linalg.norm(nn - x)**2) - 1
    return dis


def exp_distance_1Dplot(xE):
    x = np.linspace(0, 1, 1000)
    e = np.zeros(x.shape)
    for i in range(x.shape[0]):
        e[i] = exponential_distance(xE, x[i])
    fig = plt.figure()
    plt.plot(x, e, 'g')
    plt.title(r'$e(x) = exp(||x-NN(x)||^2_2)$')
    plt.show()
    plt.close(fig)
    return
