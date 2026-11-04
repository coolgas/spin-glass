"""Run the numerical regression gate on AutoDL before the finite campaign."""
import argparse
import json
from pathlib import Path
import platform
import unittest

import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Validation requires CUDA; no skipped GPU checks')
    modules = ['test_self_learning_rbm', 'test_feedback_samplers',
               'test_feedback_frontier', 'test_feedback_scaling', 'test_l6_pcd_curriculum']
    suite = unittest.defaultTestLoader.loadTestsFromNames(['transformer_qmc.'+m for m in modules])
    with threadpool_limits(limits=1):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    folder = Path(__file__).parent
    record = dict(passed=result.wasSuccessful() and not result.skipped,
        tests=result.testsRun, skipped=len(result.skipped), failures=result.failures, errors=result.errors,
        torch=torch.__version__, python=platform.python_version(), gpu=torch.cuda.get_device_name(),
        source_hashes={str(p.relative_to(folder)):sha(p) for p in folder.glob('*.py')})
    record['source_hashes']['../qmc_numerics.py'] = sha(folder.parent/'qmc_numerics.py')
    # Test objects cannot be JSON serialized in a failing run.
    for key in ['failures', 'errors']:
        record[key] = [(str(test), trace) for test,trace in record[key]]
    write(args.output, record)
    print(json.dumps(record), flush=True)
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
