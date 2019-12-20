import numpy as np
from scipy import optimize
from scipy.spatial import Delaunay
from dogs import Utils

'''
 adaptiveK.py file contains the adaptive-K continuous search function designed for AlphaDOGS and DeltaDOGS. Using the package optimize
 from scipy. 
 TODO: need to fix optimization process on each domain with SNOPT package.

 The adaptive-K continuous search function has the form:
 Sc(x) = (P(x) - y0) / K*e(x):

     Sc(x):     constant-K continuous search function;
     P(x):      Interpolation function: 
                    For AlphaDOGS: regressionparameterization because the function evaluation contains noise;
                    For DeltaDOGS: interpolationparameterization;
     e(x):      The uncertainty function constructed based on Delaunay triangulation.

 Function contained:
     tringulation_search_bound:   Search for the minimizer of continuous search function over all the Delaunay simplices 
                                  over the entire domain.
     Adoptive_K_search:           Search over a specific simplex.
     AdaptiveK_search_cost:       Calculate the value of continuous search function.


'''


#################################### Adaptive K method ####################################
def tringulation_search_bound(inter_par, xi, y0, K0, ind_min):
    inf = 1e+20
    n = xi.shape[0]
    # construct Deluanay tringulation
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

    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        # if R2 != np.inf:
        if R2 < inf:
            # initialze with body center of each simplex
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Sc[ii] = (inter_par.inter_val(x) - y0) / (R2 - np.linalg.norm(x - xc) ** 2)
            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii] = inf

    # Global one
    if np.min(Sc) < 0:
        # The minimum of Sc is negative, minimize p(x) instead.
        Scp = np.zeros(tri.shape[0])
        for ii in range(tri.shape[0]):
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Scp[ii] = inter_par.inter_val(x)

        # Globally minimize p(x)
        ind = np.argmin(Scp)
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
        simplex_bnds = Utils.search_bounds(xi[:, tri[ind, :]])
        xmin, ymin = AdaptiveK_Search_p(x, inter_par, simplex_bnds)
        result = 'pmin'
    else:
        # Minimize sc(x).
        # Global one and Local one
        index = np.array([np.argmin(Sc), np.argmin(Scl)])
        xm = np.zeros((n, 2))
        ym = np.zeros(2)
        for i in range(2):
            temp_x, ym[i] = Adaptive_K_Search(xi[:, tri[index[i], :]], inter_par, y0, K0)
            xm[:, i] = np.copy(temp_x)
        ymin = np.min(ym)
        xmin = xm[:, np.argmin(ym)].reshape(-1, 1)

        if np.argmin(ym) == 0:
            result = 'global'
        else:
            result = 'local'
    return xmin, ymin, result


#################################   Minimize interpolation function  ##################################
def AdaptiveK_Search_p(x0, inter_par, bnds):
    costfun = lambda x: inter_par.inter_val(x)
    costjac = lambda x: inter_par.inter_grad(x)
    opt = {'disp': False}
    res = optimize.minimize(costfun, x0.T[0], jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    return x, y


#################################   Minimize continuous search function  ##################################
def Adaptive_K_Search(simplex, inter_par, y0, K0):
    n = simplex.shape[0]
    R2, xc = Utils.circhyp(simplex, n)
    x = np.dot(simplex, np.ones([n + 1, 1]) / (n + 1))
    costfun = lambda x: AdaptiveK_search_cost(x, inter_par, xc, R2, y0, K0)[0]
    costjac = lambda x: AdaptiveK_search_cost(x, inter_par, xc, R2, y0, K0)[1]
    opt = {'gtol': 1e-8, 'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    res = optimize.minimize(costfun, x, jac=costjac, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun
    if abs(y) > 1e-6:
        y = - 1 / y
    else:
        y = 1e+10
    return x, y


# def AdaptiveK_search_cost(x, inter_par, xc, R2, y0, K0):
#    pos_inf = np.array([[1e+15]])
#    # neg_inf to approximate gradient, Probably has problem.
#    neg_inf = -1e+15 * np.ones(x.shape)
#    x = x.reshape(-1, 1)
#    n = x.shape[0]
#    p = interpolation.interpolate_val(x, inter_par)
#    e = R2 - np.linalg.norm(x - xc) ** 2
#    # Search function value
#    if abs(e) < 1e-18:
#        M = pos_inf
#        DM = neg_inf
#        DDM = np.zeros((n, n))
#    else:
#        M = (p - y0) / K0 * e
#    #    M = (p - y0) / e
#
#        # Gradient of interpolation and uncertainty
#        gp = interpolation.interpolate_grad(x, inter_par)
#        ge = - 2 * (x - xc)
#        # Hessian of interpolation and uncertainty
#        ggp = interpolation.interpolate_hessian(x, inter_par)
#        gge = -2 * np.identity(x.shape[0])
#        # Gradient of search
#        DM = gp / K0 * e - (p - y0) * K0 * ge/ (K0 * e) **2
#        # Hessian of search
#        DDM = - gge*K0/(p-y0) + K0*ge.dot(gp.T)/(p-y0)**2 + (K0*gp.dot(ge.T) + K0*e*ggp)/(p-y0)**2 - K0*e*2*(p-y0)*gp.dot(gp.T)/(p-y0)**4
#
#    # if optm method is chosen as TNC, use DM.T[0]
#    # method trust-ncg in scipy optimize can't handle constraints nor bounds.
#    return M, DM.T[0], DDM

def AdaptiveK_search_cost(x, inter_par, xc, R2, y0, K0):
    pos_inf = np.array([[1e+15]])
    # neg_inf to approximate gradient, Probably has problem.
    neg_inf = -1e+5 * np.ones(x.shape)
    x = x.reshape(-1, 1)
    n = x.shape[0]
    p = inter_par.inter_val(x)
    e = R2 - np.linalg.norm(x - xc) ** 2
    # Optimize - K0 * e(x) / (p(x) - y0); Extreme point doesnt change.
    if abs(p - y0) < 1e-6:
        M = np.array([[0]])
        DM = np.zeros((n, 1))
    #            DDM = np.zeros((n, n))
    else:
        M = - e * K0 / (p - y0)
        # M = (p - y0) / e
        gp = inter_par.inter_grad(x)
        ge = - 2 * (x - xc)
        #            gge = - 2 * w.T
        #            ggp = interpolation.interpolate_hessian(x, inter_par)
        DM = - ge * K0 / (p - y0) + K0 * e * gp / (p - y0) ** 2
    #            DDM = -gge * K0/(p-y0) + K0*ge.dot(gp.T)/(p-y0)**2 + (K0*ge.dot(gp.T)+e*K0*ggp)/(p-y0)**2 - (e*K0*2*(p-y0)*np.dot(gp, gp.T))/(p-y0)**4

    # method trust-ncg in scipy optimize can't handle constraints nor bounds.
    # if optm method is chosen as TNC, use DM.T[0]
    return M, DM.T[0]
