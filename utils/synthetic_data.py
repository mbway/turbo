import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # prettify matplotlib
import sklearn.gaussian_process as gp

import sys
if sys.version_info[0] == 3: # python 3
    from queue import Empty
    from math import isclose, inf
elif sys.version_info[0] == 2: # python 2
    from Queue import Empty
    inf = float('inf')
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
else:
    print('unsupported python version')

def make2D(arr):
    ''' convert a numpy array with shape (l,) into an array with shape (l,1)
        (np.atleast_2d behaves similarly but would give shape (1,l) instead)
    '''
    return arr.reshape(-1, 1)

class Noise1D:
    '''
    create and store an array of noise which can be accessed deterministically
    with get().
    '''
    def __init__(self, xs, sigma):
        self.xs = xs
        if sigma == 0.0:
            self.noise = np.zeros(shape=(len(xs)))
        else:
            self.noise = np.random.normal(0, sigma, size=(len(xs)))
    def get(self, x):
        '''
        get the noise value for the given x value
        if x is an array of values, then apply to each value in the array.
        note: this should hold:
            np.all(self.noise == self.get(self.xs))
        '''
        if isinstance(x, np.ndarray):
            # cannot vectorize if self is an argument
            @np.vectorize
            def vec_get(x):
                return self.noise[self.get_index(x)]
            return vec_get(x)
        else:
            return self.noise[self.get_index(x)]
    def get_index(self, x):
        '''
        get the index into self.noise which is closest to the given x value
        (based on the fact that entries in self.noise correspond to values of
        xs passed into the constructor)
        eg if xs = [1,2,3] then the index for x = 1.2 should be 0 (the index
        corresponding to x = 1)
        '''
        return np.argmin(np.abs(self.xs-x))

class Noise2D:
    '''
    create and store a grid of 2D noise which can be accessed deterministically
    with get().
    '''
    def __init__(self, xs, ys, sigma, fixed=True):
        self.xs = xs
        self.ys = ys
        self.fixed = fixed
        self.sigma = sigma
        if fixed:
            if sigma == 0.0:
                self.noise = np.zeros(shape=(len(xs), len(ys)))
            else:
                self.noise = np.random.normal(0, sigma, size=(len(xs), len(ys)))
    def get(self, x, y):
        '''
        get the noise value for the coordinates x and y.
        if x and y are meshgrids or arrays of points, then return a meshgrid or
        array of values with the noise values filled in.
        note: this should hold:
            np.all(self.noise == self.get(*np.meshgrid(self.xs, self.ys)))
        '''
        if isinstance(x, np.ndarray):
            if self.fixed:
                # cannot vectorize if self is an argument
                @np.vectorize
                def vec_get(x, y):
                    return self.noise[self.get_index(x, y)]
                return vec_get(x, y)
            else:
                assert x.shape == y.shape
                return np.random.normal(0, self.sigma, size=x.shape)
        else:
            if self.fixed:
                return self.noise[self.get_index(x, y)]
            else:
                return np.random.normal(0, self.sigma)
    def get_index(self, x, y):
        '''
        get the index into self.noise which is closest to the given x,y
        coordinate (based on the fact that entries in self.noise correspond to
        locations of xs and ys passed into the constructor)
        eg if xs = [1,2,3] then the index for x = 1.2 should be 0 (the index
        corresponding to x = 1)
        '''
        xi, yi = np.argmin(np.abs(self.xs-x)), np.argmin(np.abs(self.ys-y))
        # noise is indexed as [row, col] so x and y must be swapped
        return yi, xi


def random_covariance_matrix(d, max_variation):
    '''
        based on: https://math.stackexchange.com/a/1879937
        Generate a random covariance matrix which implies that it must be:
        - symmetric
        - positive semi-definite, meaning that all eigenvalues must be >0
            (but for practical purposes all eigenvalues must be >eps (for some small eps) instead of >0)

        Method: Use the eigendecomposition: A = P * D * P^T
        where
        - P is the horizontal stacking of the eigenvectors
        - D is a diagonal matrix with eigenvalues along the diagonal

        choose P randomly such that each eigenvector is orthogonal and has unit length (currently the algorithm isn't great)
        then choose (positive) eigenvalues (lengths of the eigenvectors)

        I found that drawing the eigenvalues from a Gaussian distribution led to more correlated results (which I wanted)
        Could also draw from uniform

        note: it turns out that covariance matrices can be represented as R^TR
        where R is an upper triangular matrix and who's elements correspond to
        the cosine and sine of the angles of the correlation (spherical
        parameterisation)
        (https://www.robots.ox.ac.uk/seminars/Extra/2012_30_08_MichaelOsborne.pdf)
        This would be a better way of creating the covariance matrices with a
        particular shape in mind
    '''
    P = np.random.uniform(0, 1, size=(d, d))
    P = sp.linalg.orth(P) # generate orthogonal basis for the given matrix
    assert P.shape == (d, d) # must be full rank (matrix spans d dimensions => needs d basis vectors)
    P /= np.linalg.norm(P, ord=2, axis=0) # normalise to make vectors unit length
    evs = np.abs(np.random.normal(max_variation/2, max_variation/2, size=(d,)))
    D = np.diag(evs.T)
    return np.matmul(P, np.matmul(D, P.T))

class GaussianMixture:
    def __init__(self, ranges, num_gaussians, weights=None):
        '''
            ranges: dict with keys: xmin, xmax, ymin, ymax, var
        '''
        self.num_gaussians = num_gaussians
        self.mus = np.random.uniform(
            [ranges['xmin'], ranges['ymin']],
            [ranges['xmax'], ranges['ymax']],
            size=(num_gaussians, 2)
        )
        self.sigmas = [random_covariance_matrix(d=2, max_variation=ranges['var']) for i in range(num_gaussians)]

        # the weight of each Gaussian towards the mixture
        if weights is None:
            self.weights = [1.0/num_gaussians]*num_gaussians # all with equal probability
        else:
            self.weights = weights

    def sample(self, num_samples):
        ''' draw x,y samples from the distribution '''
        samples = np.empty(shape=(num_samples, 2))
        for n in range(num_samples):
            # choose which Gaussians to sample based on their weight
            i = np.random.choice(range(self.num_gaussians), p=self.weights)
            x,y = np.random.multivariate_normal(self.mus[i], self.sigmas[i])
            samples[n,:] = (x, y)
        return samples

    def pdf(self, xy):
        ''' calculate the probability density of the distribution at a given point '''
        return sum([
            self.weights[i] * sp.stats.multivariate_normal.pdf(xy, mean=self.mus[i], cov=self.sigmas[i])
            for i in range(self.num_gaussians)])



class Data2D:
    def __init__(self):
        # ranges for selecting the means and variances
        ranges = {
            'xmin' : -10,
            'xmax' : 10,
            'ymin' : -10,
            'ymax' : 10,
            'var'  : 4   # absolute value
        }
        var = ranges['var']
        self.extent = [ranges['xmin']-var, ranges['xmax']+var, ranges['ymin']-var, ranges['ymax']+var]
        self.G = GaussianMixture(ranges, num_gaussians=5)

        self.xys = np.array(self.G.sample(2000))
        self.zs = make2D(np.apply_along_axis(self.G.pdf, 1, self.xys))

        res = 100 # resolution
        x, y = np.meshgrid(np.linspace(ranges['xmin']-var, ranges['xmax']+var, res),
                           np.linspace(ranges['ymin']-var, ranges['ymax']+var, res))
        self.all_xys = np.empty(x.shape + (2,))  # eg (100, 100) + (2,) == (100, 100, 2)
        self.all_xys[:, :, 0] = x
        self.all_xys[:, :, 1] = y
        self.x, self.y = x, y
        self.all_xys_coords = np.hstack([make2D(x.flatten()), make2D(y.flatten())]) # eg (10000, 2)

        p = lambda x,y: self.G.pdf(np.array((x,y)))
        self.all_zs = np.vectorize(p)(x, y)

    def plot_samples(self):
        plt.figure(figsize=(16,8))
        plt.axes().set_ylim(-15, 15)
        plt.axes().set_xlim(-30, 30)
        plt.plot(self.xys[:,0], self.xys[:,1], 'b.', markersize=3.0, label='data')
        plt.margins(0.1, 0.1)
        plt.legend(loc='upper left')
        plt.show()

    def plot_pdf(self):
        plt.figure(figsize=(16,8))
        plt.axes().set_ylim(-15, 15)
        plt.axes().set_xlim(-30, 30)
        plt.axes().grid(False)

        plt.imshow(self.all_zs, cmap='plasma', interpolation='nearest', origin='lower', extent=self.extent)
        plt.colorbar()
        plt.show()

    def plot_histogram(self):
        plt.figure(figsize=(16,8))
        plot_range = [[-30, 30], [-15, 15]]
        plt.axes().grid(False)
        plt.hist2d(self.xys[:,0], self.xys[:,1], bins=(128, 64), range=plot_range, cmap='plasma')
        plt.colorbar()
        plt.show()


class Data1D:
    # some functions to fit
    @staticmethod
    def fun_1(x):
        s = 0.05 # noise std dev
        y = np.cos(2*x - 1/2.0)/2.0 + np.cos(x) + 1
        return s,y
    @staticmethod
    def fun_2(x):
        s = 0.3 # noise std dev
        y = x * np.cos(x)
        return s, y

    def __init__(self, f):
        nfac = 1 # factor of n to _actually_ use (this is a hack because the
                   # keep_ids are based on the assumption of 2000 samples)
        n = 1666 * nfac # num samples (including the samples which will be thrown out)

        self.min_x = 0
        self.max_x = 10

        self.full_x = make2D(np.linspace(self.min_x, self.max_x, n))
        self.s, self.full_exact_y = f(self.full_x)

        # remove some chunks
        self.keep_ids = [(200,400), (750,1000), (1250,1450)]
        self.keep_ids = [(int(a*nfac), int(b*nfac)) for a, b in self.keep_ids]
        def remove_chunks(arr):
            return np.vstack(arr[a:b] for a, b in self.keep_ids)

        self.x = remove_chunks(self.full_x)
        self.exact_y = remove_chunks(self.full_exact_y)

        self.noise = make2D(np.random.normal(0, self.s, self.x.shape[0]))
        self.y = self.exact_y + self.noise

        # 0.0 where no samples, 1.0 where there are
        self.populated = np.zeros(len(self.full_x))
        for a, b in self.keep_ids:
            self.populated[a:b] = 1.0

        print('training GP')
        kernel = 1.0 * gp.kernels.Matern(nu=1.5) + gp.kernels.WhiteKernel()
        gp_params = dict(
            alpha = 1e-10, # noise level
            kernel = kernel.clone_with_theta(np.array([5.0566808,  1.94789113, -2.29135835])),
            optimizer = None,
            #n_restarts_optimizer = 10,
            normalize_y = True
        )
        self.gp_model = gp.GaussianProcessRegressor(**gp_params)
        self.gp_model.fit(self.x, self.y)
        self.gp_mus, self.gp_sigmas = self.gp_model.predict(self.full_x, return_std=True)
        self.gp_sigmas = make2D(self.gp_sigmas)
        print('done')

        # Jeremy says this has no theoretical grounding :(
        '''
        # larger noise
        big_noise = make2D(np.random.normal(0, self.s*4, len(self.full_x)))
        small_noise = make2D(np.random.normal(0, self.s, len(self.full_x)))
        self.big_noise = [small_noise[i] if self.populated_b[i] else big_noise[i] for i in range(len(self.full_x))]
        self.full_noisy_y = self.full_exact_y + self.big_noise
        '''


    def plot_samples(self, show_samples=True, show_exact=True, show_populated=True, show_gp=True):
        plt.figure(figsize=(16,8))
        if show_samples:
            plt.plot(self.x, self.y, 'b.', label='data')
        if show_exact:
            plt.plot(self.full_x, self.full_exact_y, 'g-', label='generator')
        if show_populated:
            plt.plot(self.full_x, self.populated, 'r-', label='populated')
        if show_gp:
            plt.plot(self.full_x, self.gp_mus, 'm-', label='gp mean')
            n_sigma = 2
            s = n_sigma*self.gp_sigmas.flatten()
            m = self.gp_mus.flatten()
            plt.fill_between(self.full_x.flatten(), m-s, m+s, alpha=0.3,
                         color='mediumpurple', label='gp ${}\\sigma$'.format(n_sigma))
        plt.margins(0.1, 0.1)
        plt.legend(loc='upper left')
        plt.show()

