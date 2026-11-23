import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import time
import json
from unittest.mock import patch
from rydberg_kagome.queue import storage_capacity
from rydberg_kagome.results import checkpoint,load_checkpoint
from rydberg_kagome.storage import BoundedLog,LOG_LIMIT


class StorageTests(unittest.TestCase):
    def test_deadline_checks_identity_and_keeps_restart(self):
        from rydberg_kagome.queue import enforce_deadline,process_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'resume.json').write_text('{}')
            worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
            rec=dict(pid=worker.pid,birth=process_identity(worker.pid),started=time.time()-10,attempt=1)
            try:
                enforce_deadline(dict(rec,birth='wrong'),{'hard_seconds':1},root)
                self.assertIsNone(worker.poll())
                enforce_deadline(rec,{'hard_seconds':1},root)
                self.assertEqual(worker.wait(timeout=5),-9)
                self.assertTrue((root/'hard-timeout-attempt1.json').exists())
                self.assertTrue((root/'resume.json').exists())
            finally:
                if worker.poll() is None: worker.kill(); worker.wait()

    def test_missing_completed_artifact_is_reported_as_corrupt(self):
        from rydberg_kagome.queue import completed_status
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'result.json').write_text(json.dumps({'hashes':{'missing':'abc'}}))
            self.assertEqual(completed_status(root,{}),'corrupt')

    def test_controlled_pauses_do_not_exhaust_failure_retries(self):
        from rydberg_kagome.queue import restart_status
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'resume.json').write_text('{}')
            self.assertEqual(restart_status(root,3,{}),'failed')
            (root/'worker-exit.json').write_text(json.dumps({'result':{'state':'memory_deferred'}}))
            self.assertEqual(restart_status(root,20,{}),'pending')
            (root/'memory-deferral.json').write_text(json.dumps({'forecast_bytes':60*2**30}))
            self.assertEqual(restart_status(root,20,{'memory_gib':64}),'memory_blocked')

    def test_real_workers_are_adopted_after_controller_restart(self):
        from rydberg_kagome.queue import process_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'; manifest=Path(folder)/'manifest.json'
            tasks=[dict(id=f'case{i}',adaptive=True,L=4,sample=i,model='full',delta=2.,
                omega=.03,chi=2,dimensions=[4,8],pairs=2000,max_sweeps=120,memory_gib=8,
                final_state_retention='analysis',checkpoint_interval=30) for i in range(3)]
            manifest.write_text(json.dumps(tasks))
            command=[sys.executable,'-B','-m','rydberg_kagome.queue','--manifest',str(manifest),
                     '--output',str(root),'--workers','2']
            with (Path(folder)/'controller.log').open('w') as log:
                controller=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+30
                while not (root/'status.json').exists() and time.monotonic()<deadline: time.sleep(.1)
                self.assertTrue((root/'status.json').exists())
                status=json.loads((root/'status.json').read_text())
                self.assertEqual(len(status['active']),2)
                identities=[(r['pid'],r['birth']) for r in status['active'].values()]
                for pid,birth in identities:
                    deadline=time.monotonic()+10
                    while True:
                        limit=next(line.split() for line in Path(f'/proc/{pid}/limits').read_text().splitlines()
                                   if line.startswith('Max address space'))
                        if limit[3]==str(8*2**30) or time.monotonic()>deadline: break
                        time.sleep(.05)
                    self.assertEqual(int(limit[3]),8*2**30)
                controller.terminate(); controller.wait(timeout=10)
                for pid,birth in identities: self.assertEqual(process_identity(pid),birth)
                adopted=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                self.assertEqual(adopted.wait(timeout=90),0)
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['counts'],{'completed':3})
            for task in tasks:
                self.assertEqual(json.loads((root/'launches'/f"{task['id']}.json").read_text())['attempt'],1)
                self.assertFalse((root/'cases'/task['id']/'resume.json').exists())

    def test_forecast_reserves_pending_outputs_active_cases_and_audits(self):
        g=2**30
        slots,r=storage_capacity(0,900*g,0,7680,8,audit_count=6)
        self.assertEqual(slots,8)
        self.assertEqual(r['pending_output_bytes'],7680*6*2**20)
        self.assertEqual(storage_capacity(60*g,900*g,2,100,8)[0],0)
        self.assertEqual(storage_capacity(0,129*g,0,0,8)[0],0)
        self.assertEqual(storage_capacity(0,900*g,8,0,8)[0],0)

    def test_bounded_checkpoint_preserves_previous_and_partial_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); target=root/'checkpoint.pkl.gz'
            checkpoint(target,{'previous':True}); previous=target.read_bytes()
            with patch.dict(os.environ,{'RYDBERG_CASE_ROOT':str(root)}), \
                 patch('rydberg_kagome.storage.FILE_LIMIT',1024):
                with self.assertRaisesRegex(OSError,'reservation exhausted'):
                    checkpoint(target,os.urandom(8192))
            self.assertEqual(target.read_bytes(),previous)
            self.assertEqual(load_checkpoint(target),{'previous':True})
            self.assertLessEqual((root/'checkpoint.pkl.gz.tmp').stat().st_size,1024)

    def test_log_bound_survives_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'worker.log'; log=BoundedLog(path)
            log.write('x'*(LOG_LIMIT+200)); log.flush()
            again=BoundedLog(path); again.write('more'); again.flush()
            self.assertEqual(path.stat().st_size,LOG_LIMIT)
            log.close(); again.close()


if __name__=='__main__': unittest.main()
