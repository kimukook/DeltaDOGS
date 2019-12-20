from deltaDOGS import *
import numpy as np


def schwefel(x):
    # Notice: the return value should be a one-dimension vector
    return np.matmul(-x.T*2, np.sin(np.sqrt(500*x)))[0]


if __name__ == '__main__':
    bnds = np.hstack((np.zeros((1, 1)), np.ones((1, 1))))
    options = DeltaDOGSOptions()
    options.set_option('Constant surrogate', True)
    options.set_option('Scipy solver', True)

    x = np.array([[.5, 1]])

    options.set_option('Initial sites known', True)
    options.set_option('Initial sites', x)

    options.set_option('Global minimizer known', True)
    options.set_option('Target value', -1.6759*2)
    options.set_option('Global minimizer', np.array([[0.8419], [0.8419]]))

    options.set_option('Initial mesh size', 2)
    options.set_option('Number of mesh refinement', 4)

    options.set_option('Function evaluation cheap', True)
    options.set_option('Plot saver', True)
    options.set_option('Candidate distance summary', True)
    options.set_option('Candidate objective value summary', True)
    options.set_option('Iteration summary', True)
    options.set_option('Optimization summary', True)

    opt = DeltaDOGS(bnds, schwefel, options)
    xmin = opt.deltadogs_optimizer()
