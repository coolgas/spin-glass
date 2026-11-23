import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from export_snapshot import export,sha


class ExportTests(unittest.TestCase):
    def fixture(self,root):
        tasks=[]
        for sid in (0,1):
            for i,g in enumerate(np.geomspace(.03,10.,30)):
                for delta in (2.,3.15):
                    task=dict(id=f's{sid}_d{delta}_g{i}',sample=sid,model='full',delta=delta,
                        omega=float(g),chi=12,L=6,adaptive=True,pairs=100)
                    tasks.append(task); folder=root/'cases'/task['id']; folder.mkdir(parents=True)
                    counts=np.zeros(109,dtype=np.int64)
                    counts[0 if sid==0 else 27]=50; counts[108 if sid==0 else 81]=50
                    np.savez(folder/'pq.npz',q=np.linspace(-1,1,109),counts=counts,p=counts/100)
                    result=dict(inputs=dict(L=6,sample_id=sid,model='full',delta=delta,omega=float(g),chi=12,pairs=100),
                        completed=True,solver_pass=sid==0,sampling_moment_check=True,rg_converged=False,
                        production_qualified=False,production_selected=True,bond_dimension_verified=False,
                        endpoints=[dict(energy=-1,D=256,sweeps=120,observables=dict(qea=.2,q2=.3))],
                        start_comparison=dict(passed=True),geometry_sha256='fixture',
                        decisions=[dict(sweep=20,next_D=256)],hashes={'pq.npz':sha(folder/'pq.npz')})
                    (folder/'result.json').write_text(json.dumps(result))
        (root/'tasks.json').write_text(json.dumps(tasks)); return tasks

    def test_equal_weight_signed_histograms_binder_and_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'; self.fixture(root)
            result=export(root,Path(temp)/'snapshot')
            self.assertEqual(result['case_count'],120)
            with np.load(Path(temp)/'snapshot'/'histograms.npz') as data:
                self.assertEqual(data['counts'].shape,(2,1,2,30,109))
                np.testing.assert_allclose(data['binder'],.5*(3-(.5*(1+.5**4))/(.5*(1+.5**2))**2))
                self.assertFalse(data['passed'][1].any())
                self.assertTrue(data['production_selected'].all())
                self.assertFalse(data['rg_converged'].any())

    def test_truncated_manifest_rejected_and_partial_disorder_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'; tasks=self.fixture(root)
            (root/'tasks.json').write_text(json.dumps(tasks[:-1]))
            with self.assertRaisesRegex(ValueError,'Incomplete parameter grid'): export(root,Path(temp)/'bad')
            (root/'tasks.json').write_text(json.dumps(tasks))
            (root/'cases'/tasks[-1]['id']/'result.json').unlink()
            result=export(root,Path(temp)/'snapshot')
            self.assertEqual(result['sample_ids'],[0]); self.assertEqual(result['case_count'],60)


if __name__=='__main__': unittest.main()
