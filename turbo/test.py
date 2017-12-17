
import sys
sys.path.append('..')

import turbo as tb
import turbo.modules as tm

def objective(x, y):
    return x**2 + y

bounds = [
    ('x', -10, 10),
    ('y', 1, 5)
]

op = tb.Optimiser(objective, 'min', bounds)
op.latent_space = tm.FixedWarpingLatentSpace(warped_params=['y'])
op.plan = tm.Plan(pre_phase_trials=3)
op.pre_phase_select = tm.random_selector()
op.maximise_acq = tm.random_quasi_newton(num_random=100, grad_restarts=5)
op.async_eval = None

op.surrogate_factory = tm.SciKitGPSurrogateFactory()
op.acq_func_factory = tm.UCB_AcquisitionFactory(beta=lambda it: 0.5)

op.run(10)

print(op.rt.trial_xs)
print(op.rt.trial_ys)

