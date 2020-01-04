#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:19:28 2017

@author: mousse
"""
import numpy as np

'''
Utils.py is implemented to generate results for AlphaDOGS and DeltaDOGS containing the following functions:
    
    bounds          :       Generate vertices under lower bound 'bnd1' and upper bound 'bnd2' for 'n' dimensions;
    mindis          :       Generate the minimum distances and corresponding point from point x to set xi;
    circhyp         :       Generate hypercircumcircle for Delaunay simplex, return the circumradius and center;
    normalize_bounds:       Normalize bounds for Delta DOGS optm solver;
    physical_bounds :       Retransform back to the physical bounds for function evaluation;
    
    search_bounds   :       Generate bounds for the Delaunay simplex during the continuous search minimization process;
    hyperplane_equations:   Determine the math expression for the hyperplane;
    
    fun_eval        :       Evaluate the function value at the given parameters via transforming it from normalized 
                            bounds to the physical bounds;
    random_initial  :       Randomly generate the initial points.
    
    
'''
################################# Utils ####################################################


def bounds(bnd1, bnd2, n):
    #   find vertex of domain for a box domain.
    #   INPUT: n: dimension, bnd1: lower bound, bnd2: upper bound.
    #   OUTPUT: vertex of domain. 2^n number vector of n-D.
    #   Example:
    #           n = 3
    #           bnd1 = np.zeros((n, 1))
    #           bnd2 = np.ones((n, 1))
    #           bnds = bounds(bnd1,bnd2,n)
    #   Author: Shahoruz Alimohammadi
    #   Modified: Dec., 2016
    #   DELTADOGS package
    assert bnd1.shape == (n, 1) and bnd2.shape == (n, 1), 'lb(bnd1) and ub(bnd2) should be 2 dimensional vector.'
    bnds = np.kron(np.ones((1, 2 ** n)), bnd2)
    for ii in range(n):
        tt = np.mod(np.arange(2 ** n) + 1, 2 ** (n - ii)) <= 2 ** (n - ii - 1) - 1
        bnds[ii, tt] = bnd1[ii]
    return bnds


def unique_support_points(xU, xE):
    """
    This function deletes the the elements from xU that is repeated in xE.
    :param xU:  Support points.
    :param xE:  Evaluated points.
    :return:
    """
    i = 0
    size = xU.shape[1]
    while i < size:
        if mindis(xU[:, i], xE)[0] < 1e-10:
            xU = np.delete(xU, i, 1)
            i = max(i-1, 0)
            size -= 1
        else:
            i += 1
    return xU


def mindis(x, xi):
    '''
    calculates the minimum distance from all the existing points
    :param x: x the new point
    :param xi: xi all the previous points
    :return: [ymin ,xmin ,index]
    '''
    x = x.reshape(-1, 1)
    assert xi.shape[1] != 0, 'The set of Evaluated points has size 0!'
    dis = np.linalg.norm(xi-x, axis=0)
    val = np.min(dis)
    idx = np.argmin(dis)
    xmin = xi[:, idx].reshape(-1, 1)
    return val, idx, xmin


def circhyp(x, N):
    # circhyp     Circumhypersphere of simplex
    #   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
    #   and the square of the radius of the N-dimensional hypersphere
    #   encircling the simplex defined by its N+1 vertices.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    #   DOGS package

    test = np.sum(np.transpose(x) ** 2, axis=1)
    test = test[:, np.newaxis]
    m1 = np.concatenate((np.matrix((x.T ** 2).sum(axis=1)), x))
    M = np.concatenate((np.transpose(m1), np.matrix(np.ones((N + 1, 1)))), axis=1)
    a = np.linalg.det(M[:, 1:N + 2])
    c = (-1.0) ** (N + 1) * np.linalg.det(M[:, 0:N + 1])
    D = np.zeros((N, 1))
    for ii in range(N):
        M_tmp = np.copy(M)
        M_tmp = np.delete(M_tmp, ii + 1, 1)
        D[ii] = ((-1.0) ** (ii + 1)) * np.linalg.det(M_tmp)
        # print(np.linalg.det(M_tmp))
    # print(D)
    xC = -D / (2.0 * a)
    #	print(xC)
    R2 = (np.sum(D ** 2, axis=0) - 4 * a * c) / (4.0 * a ** 2)
    #	print(R2)
    return R2, xC


def normalize_bounds(x0, lb, ub):
    """
    Compute the normalized data points, between [0, 1]
    :param x0:  Data points in real-world
    :param lb:  Physical lower bound
    :param ub:  Physical upper bound
    :return:
    """
    n = len(lb)  # n represents dimensions
    m = x0.shape[1]  # m represents the number of sample data
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j] - lb[i]) / (ub[i] - lb[i])
    return x


def physical_bounds(x0, lb, ub):
    """
    :param x0: normalized point
    :param lb: real lower bound
    :param ub: real upper bound
    :return: physical scale of the point
    """
    n = len(lb)  # n represents dimensions
    try:
        m = x0.shape[1]  # m represents the number of sample data
    except:
        m = x0.shape[0]
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j])*(ub[i] - lb[i]) + lb[i]

    return x


def search_bounds(xi):
    n = xi.shape[0]
    srch_bnd = []
    for i in range(n):
        rimin = np.min(xi[i, :])
        rimax = np.max(xi[i, :])
        temp = (rimin, rimax)
        srch_bnd.append(temp)
    simplex_bnds = tuple(srch_bnd)
    return simplex_bnds


def search_simplex_bounds(xi):
    '''
    Return the n+1 constraints defined by n by n+1 Delaunay simplex xi.
    The optimization for finding minimizer of Sc should be within the Delaunay simplex.
    :param xi: xi should be (n) by (n+1). Each column denotes a data point.
    :return: Ax >= b constraints.
    A: n+1 by n
    b: n+1 by 1
    '''
    # TODO fix when dim = 1.
    n = xi.shape[0]  # dimension of input
    m = xi.shape[1]  # number of input, should be exactly the same as n+1.
    # The linear constraint, which is the boundary of the Delaunay triangulation simplex.
    A = np.zeros((m, n))
    b = np.zeros((m, 1))
    for i in range(m):
        direction_point = xi[:, i].reshape(-1, 1)  # used to determine the type of inequality, <= or >=
        plane_points = np.delete(xi, i, 1)  # should be an n by n square matrix.
        A[i, :], b[i, 0] = hyperplane_equations(plane_points)
        if np.dot(A[i, :].reshape(-1, 1).T, direction_point) < b[i, :]:
            # At this point, the simplex stays at the negative side of the equation, assign minus sign to A and b.
            A[i, :] = np.copy(-A[i, :].reshape(-1, 1).T)
            b[i, 0] = np.copy(-b[i, :])
    return A, b


def hyperplane_equations(points):
    '''
    Return the equation of n points hyperplane in n dimensional space.

    Reference website:
    https://math.stackexchange.com/questions/2723294/how-to-determine-the-equation-of-the-hyperplane-that-contains-several-points

    :param points: Points is an n by n square matrix. Each column represents a data point.
    :return: A and b (both 2 dimensional array) that satisfy Ax = b.
    '''
    n, m = points.shape  # n dimension of points. m should be the same as n
    base_point = points[:, -1].reshape(-1, 1)
    matrix = (points - base_point)[:, :-1].T  # matrix should be n-1 by n, each row represents points - base_point.
    A = np.zeros((1, n))
    b = np.zeros((1, 1))
    for j in range(n):
        block = np.delete(matrix, j, 1)  # The last number 1, denotes the axis. 1 is columnwise while 0 is rowwise.
        A[0, j] = (-1) ** (j+1) * np.linalg.det(block)
    b[0, 0] = np.dot(A, base_point)
    return A, b


def fun_eval(fun, lb, ub, x):
    x = x.reshape(-1, 1)
    x_phy = physical_bounds(x, lb, ub)
    y = fun(x_phy)
    return y


def random_initial(n, m, Nm, Acons, bcons):
    xU = bounds(np.zeros((n, 1)), np.ones((n, 1)), n)
    xE = np.empty(shape=[n, 0])
    while xE.shape[1] < m:
        temp = np.random.rand(n, 1)
        temp = np.round(temp * Nm) / Nm
        dis1, _, _ = mindis(temp, xU)
        if dis1 > 1e-6 and (np.dot(Acons, temp) - bcons <= 0).all():
            if xE.shape[1] == 0:
                xE = np.hstack(( xE, temp ))
            else:
                dis2, _, _ = mindis(temp, xE)
                if dis2 > 1e-6:
                    xE = np.hstack(( xE, temp ))
        else:
            continue
    return xE
