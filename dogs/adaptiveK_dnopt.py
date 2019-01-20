from   optimize      import dnopt, DNOPT_options
import numpy         as np
from   scipy.spatial import Delaunay
from   dogs          import Utils
from   dogs          import interpolation
import scipy.io      as io
'''
 adaptiveK_snopt.py file contains functions used for DeltaDOGS(Lambda) algorithm.
 Using the package optimize(DNOPT) provided by Prof. Philip Gill and Dr. Elizabeth Wong, UCSD.


 The adaptive-K continuous search function has the form:
 Sc(x) = (P(x) - y0) / (K * e(x)):

     Sc(x):     constant-K continuous search function;
     P(x):      Interpolation function: 
                    For AlphaDOGS: regressionparameterization because the function evaluation contains noise;
                    For DeltaDOGS: interpolationparameterization;
     e(x):      The uncertainty function constructed based on Delaunay triangulation.

 Function contained:
     tringulation_search_bound_dnopt:   Search for the minimizer of continuous search function over all the Delaunay simplices 
                                        over the entire domain.
     Adaptive_K_Search_dnopt:                 Search over a specific simplex.
     AdaptiveK_search_cost:             Calculate the value of continuous search function.


'''
##################################  adaptive K search SNOPT ###################################


def trisearch_bound_dnopt(inter_par, xi, y0, K0, ind_min):
    # reddir is a vector
    inf = 1e+20
    n = xi.shape[0]  # The dimension of the reduced model.

    # 0: Build up the Delaunay triangulation based on reduced subspace.
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]
    # Sc contains the continuous search function value of the center of each Delaunay simplex

    # 1: Identify the minimizer of adaptive K continuous search function
    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        if R2 < inf:
            # initialize with body center of each simplex
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Sc[ii] = (interpolation.interpolate_val(x, inter_par) - y0) / (R2 - np.linalg.norm(x - xc) ** 2)
            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii] = inf

    if np.min(Sc) < 0:
        func = 'p'
        # The minimum of Sc is negative, minimize p(x) instead.
        Scp = np.zeros(tri.shape[0])
        Scpl = np.zeros(tri.shape[0])
        for ii in range(tri.shape[0]):
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Scp[ii] = interpolation.interpolate_val(x, inter_par)
            if np.sum(ind_min == tri[ii, :]):
                Scpl[ii] = np.copy(Scp[ii])
            else:
                Scpl[ii] = inf
        else:
            Scpl[ii] = inf
            Scp[ii] = inf
        # Globally minimize p(x)
        ind = np.argmin(Scp)
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
        simplex_bnds = Utils.search_bounds(xi[:, tri[ind, :]])
        xm, ym = AdaptiveK_Search_p_dnopt(x, inter_par, simplex_bnds)
        # Locally minimize p(x)
        ind = np.argmin(Scpl)
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
        simplex_bnds = Utils.search_bounds(xi[:, tri[ind, :]])
        xml, yml = AdaptiveK_Search_p_dnopt(x, inter_par, simplex_bnds)

    else:
        func = 'sc'
        # Global one, the minimum of Sc has the minimum value of all circumcenters.
        ind = np.argmin(Sc)
        R2, xc = Utils.circhyp(xi[:, tri[ind, :]], n)
        # x is the center of this simplex
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
        # First find minimizer xr on reduced model, then find the 2D point corresponding to xr. Constrained optm.
        simplex_bnds = Utils.search_bounds(xi[:, tri[ind, :]])
        xm, ym = AdaptiveK_Search_dnopt(x, inter_par, xc, R2, y0, K0, simplex_bnds)
        # Local one
        ind = np.argmin(Scl)
        R2, xc = Utils.circhyp(xi[:, tri[ind, :]], n)
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
        simplex_bnds = Utils.search_bounds(xi[:, tri[ind, :]])
        xml, yml = AdaptiveK_Search_dnopt(x, inter_par, xc, R2, y0, K0, simplex_bnds)

    if yml < ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
        result = 'local'
    else:
        result = 'glob'
    xm = xm.reshape(-1, 1)
    return xm, ym, result, func


#################################   Minimize interpolation function  ##################################
def AdaptiveK_Search_p_dnopt(x0, inter_par, bnds):
    n = x0.shape[0]
    x0 = x0.T[0]
    save_opt_for_dnopt_p(inter_par)

    bl = np.zeros(n, float)
    bu = np.ones(n, float)
    H = np.zeros((n, n))
    for i in range(n):
        bl[i] = bnds[i][0]
        bu[i] = bnds[i][1]

    options = DNOPT_options()
    inf = 1.0e+20
    options.setOption('Infinite bound', inf)
    options.setOption('Verify level', 3)
    options.setOption('Verbose', False)
    options.setOption('Print filename', 'DDOGS.out')

    result = dnopt(dogsobj_p, None, nnObj=n, nnCon=0, nnJac=0, x0=x0, H=H, name='DeltaDOGS_dnopt', iObj=0, bl=bl,
                   bu=bu, options=options)
    x = result.x
    y = result.objective
    return x.reshape(-1, 1), y


def save_opt_for_dnopt_p(inter_par):
    var_opt = {}
    if inter_par.method == "NPS":
        var_opt['inter_par_method'] = inter_par.method
        var_opt['inter_par_w'] = inter_par.w
        var_opt['inter_par_v'] = inter_par.v
        var_opt['inter_par_xi'] = inter_par.xi
    io.savemat("opt_info.mat", var_opt)
    return


def dogsobj_p(mode, x, fObj, gObj, nState):
    fObj = 0.0
    if mode == 0 or mode == 2:
        fObj = adaptivek_search_cost_dr_dnopt_p(x)[0]
    if mode == 0:
        return mode, fObj
    if mode == 1 or mode == 2:
        gObj = adaptivek_search_cost_dr_dnopt_p(x)[1]

    return mode, fObj, gObj


def adaptivek_search_cost_dr_dnopt_p(x):
    x = x.reshape(-1, 1)
    var_opt = io.loadmat("opt_info.mat")

    method = var_opt['inter_par_method'][0]
    inter_par = interpolation.Inter_par(method=method)
    inter_par.w = var_opt['inter_par_w']
    inter_par.v = var_opt['inter_par_v']
    inter_par.xi = var_opt['inter_par_xi']
    p = interpolation.interpolate_val(x, inter_par)
    gp = interpolation.interpolate_grad(x, inter_par)
    return p, gp.T[0]


#################################   Minimize continuous search function  ##################################
def AdaptiveK_Search_dnopt(x0, inter_par, xc, R2, y0, K0, bnds):
    # Find the minimizer of the search fucntion in a simplex using SNOPT package.
    n = x0.shape[0]
    x0 = x0.T[0]
    save_opt_for_dnopt(n, inter_par, xc, R2, y0, K0)

    bl = np.zeros(n, float)
    bu = np.ones(n, float)
    H = np.zeros((n, n))
    for i in range(n):
        bl[i] = bnds[i][0]
        bu[i] = bnds[i][1]

    options = DNOPT_options()
    inf = 1.0e+20
    options.setOption('Infinite bound', inf)
    options.setOption('Verify level', 3)
    options.setOption('Verbose', False)
    options.setOption('Print filename', 'DDOGS.out')

    result = dnopt(dogsobj, None, nnObj=n, nnCon=0, nnJac=0, x0=x0, H=H, name='DeltaDOGS_dnopt', iObj=0, bl=bl,
                   bu=bu, options=options)

    x = result.x
    y = result.objective
    if abs(y) > 1e-6:
        y = - 1 / y
    else:
        y = 1e+10
    return x.reshape(-1, 1), y


def save_opt_for_dnopt(n, inter_par, xc, R2, y0, K0):
    var_opt = {}
    if inter_par.method == "NPS":
        var_opt['inter_par_method'] = inter_par.method
        var_opt['inter_par_w'] = inter_par.w
        var_opt['inter_par_v'] = inter_par.v
        var_opt['inter_par_xi'] = inter_par.xi
    var_opt['n'] = n
    var_opt['xc'] = xc
    var_opt['R2'] = R2
    var_opt['y0'] = y0
    var_opt['K0'] = K0
    io.savemat("opt_info.mat", var_opt)
    return


def adaptivek_search_cost_dr_dnopt(x):
    x = x.reshape(-1, 1)
    var_opt = io.loadmat("opt_info.mat")

    n = var_opt['n'][0, 0]
    xc = var_opt['xc']
    R2 = var_opt['R2'][0, 0]
    K0 = var_opt['K0'][0, 0]
    y0 = var_opt['y0'][0, 0]
    method = var_opt['inter_par_method'][0]
    inter_par = interpolation.Inter_par(method=method)
    inter_par.w = var_opt['inter_par_w']
    inter_par.v = var_opt['inter_par_v']
    inter_par.xi = var_opt['inter_par_xi']
    p = interpolation.interpolate_val(x, inter_par)
    e = R2 - np.linalg.norm(x - xc) ** 2
    gp = interpolation.interpolate_grad(x, inter_par)
    ge = - 2 * (x - xc)
    if abs(p-y0) < 1e-6:
        de = 1e-6
    else:
        de = p - y0
    M = - K0 * e / de
    DM = - K0 * ge / de + K0 * e * gp / de ** 2
    return M, DM.T[0]


def dogsobj(mode, x, fObj, gObj, nState):
    fObj = 0.0
    if mode == 0 or mode == 2:
        fObj = adaptivek_search_cost_dr_dnopt(x)[0]
    if mode == 0:
        return mode, fObj
    if mode == 1 or mode == 2:
        gObj = adaptivek_search_cost_dr_dnopt(x)[1]

    return mode, fObj, gObj


