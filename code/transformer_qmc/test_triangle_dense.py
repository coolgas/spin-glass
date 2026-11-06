"""Regression checks for gated dispatch and controller handover."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import triangle_overlap_dense as dense
from . import triangle_correlation_parallel as parallel
from .triangle_correlation_dmrg import write


class DenseDispatchTests(unittest.TestCase):
    def test_memory_headroom_counts_only_reclaimable_inactive_file_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'memory.max').write_text(str(100*2**30))
            (root/'memory.current').write_text(str(60*2**30))
            (root/'memory.stat').write_text(
                f'file {25*2**30}\nshmem {2**30}\ninactive_file {12*2**30}\n')
            with patch.object(parallel, 'Path', return_value=root):
                self.assertEqual(parallel.memory_headroom(), 52.)
            # Shared memory is not reclaimable file cache.
            (root/'memory.stat').write_text(
                f'file {25*2**30}\nshmem {20*2**30}\ninactive_file {12*2**30}\n')
            with patch.object(parallel, 'Path', return_value=root):
                self.assertEqual(parallel.memory_headroom(), 45.)

    def test_matched_pilots_gate_each_size(self):
        for size in (4, 6, 8):
            complete = {parallel.key(dense.task(4, c)) for c in (3, 16)}
            pilots = [dense.task(size, c, g, s) for c in (3, 16)
                      for g in (.1, .5, 1.) for s in (60000, 60001)]
            complete.update(map(parallel.key, pilots))
            job = dict(mode='case', task=dense.task(size, 3, .125))
            passed = lambda t, exact=False: parallel.key(t) in complete
            self.assertEqual(dense.dependencies(job, passed, lambda t: True), [])
            for pilot in pilots:
                k = parallel.key(pilot)
                complete.remove(k)
                self.assertTrue(dense.dependencies(job, passed, lambda t: True))
                complete.add(k)
            self.assertTrue(dense.dependencies(job, passed, lambda t: False))

    def test_scope_and_unique_jobs(self):
        jobs = dense.jobs()
        self.assertEqual(len(jobs), 2742)
        self.assertEqual(len({dense.name(j) for j in jobs}), len(jobs))
        self.assertEqual({j['task']['L'] for j in jobs}, {4, 6, 8})

    def test_failed_gate_terminates_without_retry_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root/'backend.json'
            write(backend, dict(validated=True, selected='cuda'))
            gate = dict(mode='case', task=dense.task(4, 16))
            profile = dict(mode='profile', task=dense.task(8, 16))
            write(root/'attempts.json', {dense.name(gate): dict(outcome='unresolved')})
            args = SimpleNamespace(output=root, reuse=[], backend=backend,
                                   profile_threads=2, dispatch_lock=root/'dispatch.lock')
            with patch.object(dense, 'jobs', return_value=[gate, profile]), \
                    patch.object(dense.subprocess, 'Popen') as spawn:
                dense.controller(args)
                spawn.assert_not_called()
            state = json.loads((root/'status.json').read_text())
            self.assertEqual(state['state'], 'complete_with_unresolved_cases')
            self.assertIn(dense.name(profile), state['blocked_dependencies'])
            with args.dispatch_lock.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_external_reservations_prevent_oversubscription(self):
        class StopLoop(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root/'backend.json', dict(validated=True, selected='cuda'))
            args = SimpleNamespace(output=root, reuse=[], backend=root/'backend.json',
                                   profile_threads=2, dispatch_lock=root/'dispatch.lock')
            peer = dict(pid=os.getpid(), task=dense.task(8, 3), threads=16,
                        memory_gib=2., rss_gib=1., disk_gib=.1)
            with patch.object(dense, 'jobs', return_value=[dict(mode='case', task=dense.task(4, 16))]), \
                    patch.object(dense, 'external_workers', return_value=[peer]), \
                    patch.object(dense.time, 'sleep', side_effect=StopLoop), \
                    patch.object(dense.subprocess, 'Popen') as spawn:
                with self.assertRaises(StopLoop):
                    dense.controller(args)
                spawn.assert_not_called()
            state = json.loads((root/'status.json').read_text())
            self.assertEqual(state['state'], 'waiting_for_resources')
            self.assertEqual(state['worker_cpu_threads'], 16)

    def test_peer_status_ignores_dead_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = dict(pid=os.getpid(), threads=2, memory_gib=4.)
            write(root/'status.json', dict(running=dict(live=live, dead=dict(pid=999999999))))
            self.assertEqual(parallel.peer_workers([root]), [live])


if __name__ == '__main__':
    unittest.main()
