
import optimiser as to
import modules as tm

def objective(x, y):
    return x**2 + y

bounds = [
    ('x', -10, 10),
    ('y', 0, 5)
]

op = to.Optimiser(lambda config: objective(**config), 'min', bounds)
op.latent_space = tm.FixedWarpingLatentSpace(bounds, warped_params=['y'])
op.plan = tm.Plan(pre_phase_trials=3)
op.pre_phase_select = tm.random_selector()
op.maximise_acq = tm.random_quazi_newton(random_samples=100, grad_restarts=5)
op.async_eval = None

op.surrogate_factory = SciKitSurrogateFactory(gp_params={})
op.acq_func_factory = tm.UCB_Acquisition_Factory(beta=lambda it: 0.5)

op.run(10)

