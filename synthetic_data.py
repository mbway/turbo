import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # prettify matplotlib

def make2D(arr):
    ''' convert a numpy array with shape (l,) into an array with shape (l,1)
        (np.atleast_2d behaves similarly but would give shape (1,l) instead)
    '''
    return arr.reshape(arr.shape[0], 1)

def random_covariance_matrix(d, max_variation):
    '''
        Generate a random covariance matrix which implies that it must be:
        - symmetric
        - positive semi-definate, meaning that all eigenvalues must be >0
            (but for practical purposes all eigenvalues must be >eps (for some small eps) instead of >0)

        Method: Use the eigendecomposition: A = P * D * P^T
        where
        - P is the horizontal stacking of the eigenvectors
        - D is a diagonal matrix with eigenvalues along the diagonal

        choose P randomly such that each eigenvector is orthogonal and has unit length (currently the algorithm isn't great)
        then choose (positive) eigenvalues (lengths of the eigenvectors)

        I found that drawing the eigenvalues from a Gaussian distribution led to more correlated results (which I wanted)
        Could also draw from uniform
    '''
    P = np.random.uniform(0, 1, size=(d, d))
    P = sp.linalg.orth(P) # generate orthogonal basis for the given matrix
    P /= np.linalg.norm(P, ord=2, axis=0)
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
        self.n = 2000 # num samples (including the samples which will be thrown out)

        self.min_x = 0
        self.max_x = 12

        self.full_x = make2D(np.linspace(self.min_x, self.max_x, self.n))
        self.s, self.full_exact_y = f(self.full_x)

        # remove some chunks
        def remove_chunks(arr):
            return np.vstack([arr[200:400], arr[750:1000], arr[1250:1450]])

        self.x = remove_chunks(self.full_x)
        self.exact_y = remove_chunks(self.full_exact_y)
        self.n = self.x.shape[0]

        self.noise = make2D(np.random.normal(0, self.s, self.n))
        self.y = self.exact_y + self.noise

        # populated as booleans
        self.populated_b = [False] * len(self.full_x)
        for i, x in enumerate(self.full_x):
            for dx in self.x:
                if math.isclose(x, dx, abs_tol=1e-1):
                    self.populated_b[i] = True
                    break
        self.populated = np.array([1 if p else 0 for p in self.populated_b])

    def plot_samples(self, show_populated=True):
        plt.figure(figsize=(16,8))
        plt.plot(self.x, self.y, 'b.', label='data')
        plt.plot(self.full_x, self.full_exact_y, 'g-', label='generator')
        if show_populated:
            plt.plot(self.full_x, self.populated, 'r-', label='populated')
        plt.margins(0.1, 0.1)
        plt.legend(loc='upper left')
        plt.show()
