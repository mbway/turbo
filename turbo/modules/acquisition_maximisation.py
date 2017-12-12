
from naive_selectors import random_selector

class random_quasi_newton:
    def __init__(self, random_samples=1000, grad_restarts=10, start_from_best=2):
        '''
        Args:
            random_samples: number of random points to sample to search for the
                maximum
            grad_restarts: number of restarts to allow the gradient-based
                optimiser to search for the maximum.
            start_from_best: number of points from the random phase to use
                as starting points in the gradient-based stage. Included in the
                num_grad_restarts total, the remaining points will be chosen at
                random.  should be <= num_random and <= num_grad_restarts
        '''
        self.random_samples = random_samples
        self.grad_restarts = grad_restarts
        self.start_from_best = start_from_best
        self.gen_random = random_selector()
        assert start_from_best <= random_samples
        assert start_from_best <= grad_restarts

    def __call__(self, latent_bounds, acq):
        '''
        Returns:
            x, y or None, -inf if maximisation fails.
            x: shape=(1, num_attribs): the function input which produces the smallest output
            y: float: the value of f(x)
        '''

        # maximise the function by minimising the negation of the function, then
        # negating the results at the end. This is necessary because scipy only
        # offers a gradient based minimiser.

        # keep track of the current best
        best_x = None
        best_y = inf

        # minimise by random sampling
        if num_random > 0:
            random_x = self.gen_random(num_random, latent_bounds)
            random_y = -acq(random_x)

            best_ids = np.argsort(random_y, axis=0).flatten() # sorted indices
            best_random_i = best_ids[0] # smallest
            best_x = make2D_row(random_x[best_random_i])
            best_y = random_y[best_random_i]

        # minimise by gradient-based optimiser
        if num_grad_restarts > 0:
            if num_random > 0:
                # see if gradient-based optimisation can improve upon the best
                # samples from the last stage
                # N random xs from the last step with the smallest y values
                best_starts = random_x[best_ids[:take_best_random]]
                new_starts = gen_random(num_grad_restarts - take_best_random)
                starting_points = np.vstack((best_starts, new_starts))
            else:
                starting_points = gen_random(num_grad_restarts)

            def bfgs(j):
                starting_point = make2D_row(starting_points[j])
                # the minimiser passes x as (num_attribs,) but f wants (1,num_attribs)
                neg_f = lambda x: -acq(make2D_row(x))

                # result is an OptimizeResult object
                # optimisation process may trigger warnings
                result = scipy.optimize.minimize(
                    fun=neg_f,
                    x0=starting_point,
                    bounds=bounds,
                    method='L-BFGS-B', # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
                    options=dict(maxiter=15000) # maxiter=15000 is default
                )

                if not result.success:
                    warnings.warn('restart {}/{} of gradient-based optimisation failed'.format(
                        j, num_grad_restarts))
                    return None, None
                return result.x, result.fun

            # @Performance: I was going to parallelise this, but it has reasonable
            # performance. Between 0.1s and 4s in my testing and although it does
            # increase as the optimisation progresses, the trend appears to be
            # sub-linear.
            for j in range(num_grad_restarts):
                res_x, res_y = bfgs(j)
                if res_y is not None and res_y < best_y:
                    best_x = res_x # shape=(num_attribs,)
                    best_y = res_y # shape=(1,1)

        if best_x is None:
            return None, -inf
        else:
            best_x = make2D_row(best_x) # shape=(1, num_attribs)
            # ensure that the chosen value lies within the bounds (which may not
            # be the case due to floating point error)
            low_bounds, high_bounds = zip(*bounds)
            best_x = np.clip(best_x, low_bounds, high_bounds)
            best_y = -np.asscalar(best_y) # undo negation
            return best_x, best_y
