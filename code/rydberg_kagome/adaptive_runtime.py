"""Explicit one-sweep execution and in-place TeNPy dimension transitions."""


def disable_mixer(engine):
    if engine.mixer is not None:
        engine.mixer_deactivate()
    # Deactivation alone leaves the mixer's matrix-valued Schmidt factors.
    # Cleanup preserves the state and transforms the cached environments.
    engine.mixer_cleanup()


def increase_dimension(engine, cap):
    if cap <= engine.trunc_params['chi_max']:
        raise ValueError('Dimension must increase')
    disable_mixer(engine)
    engine.trunc_params['chi_max'] = cap
    engine.options.subconfig('mixer_params')['disable_after'] = 5
    engine.mixer_activate()


def one_sweep(engine):
    before = engine.sweeps
    energy, psi = engine.run_iteration()
    if engine.sweeps != before + 1:
        raise RuntimeError('A diagnostic iteration must perform exactly one sweep')
    return energy, psi
