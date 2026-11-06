"""Run on AutoDL before handing over the matched dense campaign."""
import argparse
import json
import pickle
from pathlib import Path

from .triangle_correlation_dmrg import case, write, ResourceDeferred
from .triangle_correlation_highchi import guard, worker_limits
from .triangle_overlap_dense import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    seen = []

    def interrupt(folder, model, mpo, psi, dimension, stage):
        seen.append(stage)
        if stage in ('seed0_D64', 'seed1_D64'):
            raise ResourceDeferred('Intentional checkpoint recovery check')

    try:
        case(root, 4, .5, 60000, 2, m=4, validate=True, resume=True,
             stage_guard=interrupt, qualify_overlap=True)
    except ResourceDeferred:
        pass
    else:
        raise AssertionError('Expected both starts to defer')
    assert 'seed0_D32' in seen and 'seed1_D32' in seen, seen
    resumed = []

    def observe(folder, model, mpo, psi, dimension, stage):
        resumed.append(stage)

    result = case(root, 4, .5, 60000, 2, m=4, validate=True, resume=True,
                  stage_guard=observe, qualify_overlap=True)
    assert result['solver_passed'] and result['overlap_gates_enabled']
    assert 'seed0_D32' not in resumed and 'seed1_D32' not in resumed
    folder = root/'cases/L4_g0.5_s60000_m4_c2'
    assert audit(folder)['qualified']
    prior = json.loads(Path('/root/autodl-tmp/triangle-correlation-retention-20261105-v6/results/cases/L4_g0.5_s60000_m4_c2/result.json').read_text())
    assert abs(result['records'][-1]['energy']-prior['records'][-1]['energy'])/48 < 1e-10
    with (folder/'model.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    import os
    previous = os.environ.get('HIGHCHI_MEMORY_GIB')
    os.environ['HIGHCHI_MEMORY_GIB'] = '.01'
    try:
        try:
            guard(folder, saved['model'], saved['mpo'], None, 512, 'intentional_limit')
        except ResourceDeferred:
            pass
        else:
            raise AssertionError('Resource guard must defer')
    finally:
        if previous is None:
            os.environ.pop('HIGHCHI_MEMORY_GIB')
        else:
            os.environ['HIGHCHI_MEMORY_GIB'] = previous
    assert worker_limits({'production_estimates': {'512': {'memory_gib': 200, 'checkpoint_gib': 1}}}, 100) is None
    write(root/'passed.json', dict(passed=True, both_start_recovery=True,
          exact_retained_space_and_observables=True, unchanged_energy=True,
          resource_deferral=True))
    print('DENSE PREFLIGHT PASSED', flush=True)


if __name__ == '__main__':
    main()
