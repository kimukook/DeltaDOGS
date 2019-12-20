#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:22:25 2017

@author: mousse
"""
import numpy as np
'''
 interpolation.py contains interpolation and regression for DeltaDOGS and AlphaDOGS
    NPS( Natural Polyharmonic Spline ):
        P(x) = sum(w_i * phi(r) ) + v.T * [1; x]
        phi(r) = r**3, r = norm(x-xi)
        
        Inter_par:                          Contains the parameter w, v and xi(seldom used) for NPS, a for MAPS.
        interpolateparameterization:        Perform NPS interpolation on DeltaDOGS
        regressionparametarization:         Perform NPS regression on AlphaDOGS
        smoothing_polyharmonic
        interpolate_val:                    Calculate interpolation/regression function values in the parameter space.
        interpolate_grad:                   Calculate the gradient of interpolation/regression in the parameter space.
        interpolate_hessian:                Calculate the hessian matrix of interpolation/regression in the parameter space.
        
'''
################################## Interpolation and Regression ########################################################


class InterParams:
    def __init__(self, xE):
        self.method = "NPS"
        self.n  = xE.shape[0]
        self.m  = xE.shape[1]
        self.xi = np.copy(xE)
        self.w  = None
        self.v  = None
        self.y  = None
        self.sigma = None

    def interpolateparameterization(self, yE):
        self.w  = np.zeros((self.m, 1))
        self.v  = np.zeros((self.n + 1, 1))
        self.y  = np.copy(yE)
        if self.method == 'NPS':
            A = np.zeros((self.m, self.m))
            for ii in range(self.m):
                for jj in range(self.m):
                    A[ii, jj] = (np.dot(self.xi[:, ii] - self.xi[:, jj], self.xi[:, ii] - self.xi[:, jj])) ** (3.0 / 2.0)

            V = np.vstack((np.ones((1, self.m)), self.xi))
            A1 = np.hstack((A, np.transpose(V)))
            A2 = np.hstack((V, np.zeros((self.n + 1, self.n + 1))))
            yi = self.y[np.newaxis, :]
            b = np.concatenate([np.transpose(yi), np.zeros((self.n + 1, 1))])
            A = np.vstack((A1, A2))
            wv = np.linalg.lstsq(A, b, rcond=-1)
            wv = np.copy(wv[0])
            self.w = np.copy(wv[:self.m])
            self.v = np.copy(wv[self.m:])
            yp = np.zeros(self.m)
            for ii in range(self.m):
                yp[ii] = self.inter_val(self.xi[:, ii])
            return yp

    def inter_val(self, x):
        '''
        Calculate the value of objective interpolant at x.
        :param x:           The intended position to calculate the gradient of interpolation/regression function
        :param inter_par:   The parameter set for interpolation/regression
        return:             The interpolation/regression function values at x.
        '''
        x = x.reshape(-1, 1)
        if self.method == "NPS":
            S = self.xi - x
            return np.dot(self.v.T, np.concatenate([np.ones((1, 1)), x], axis=0))[0] + np.dot(self.w.T, (
                np.sqrt(np.diag(np.dot(S.T, S))) ** 3))

    def inter_grad(self, x):
        '''
        Calculate the gradient of interpolant at x.
        :param x:           The intended position to calculate the gradient of interpolation/regression function
        :param inter_par:   The parameter set for interpolation/regression
        return:             The column vector of the gradient information at point x.
        '''

        x = x.reshape(-1, 1)
        g = np.zeros((self.n, 1))
        if self.method == "NPS":
            for ii in range(self.m):
                X = x - self.xi[:, ii].reshape(-1, 1)
                g = g + 3 * self.w[ii] * X * np.linalg.norm(X)
            g = g + self.v[1:]
            return g

    def interpolate_hessian(self, x):
        '''
        :param x:           The intended position to calculate the gradient of interpolation/regression function
        :param inter_par:   The parameter set for interpolation/regression
        return:             The hessian matrix of at x.
        '''
        if self.method == "NPS":
            H = np.zeros((self.n, self.n))
            for ii in range(self.m):
                X = x[:, 0] - self.xi[:, ii]
                if np.linalg.norm(X) > 1e-5:
                    H = H + 3 * self.w[ii] * ((X * X.T) / np.linalg.norm(X) + np.linalg.norm(X) * np.identity(self.n))
            return H
