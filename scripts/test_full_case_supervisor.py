"""Check terminal classification using a synthetic case worker."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class FullCaseSupervisorTests(unittest.TestCase):
    def test_terminal_gate_requires_solver_and_sampling_success(self):
        for solver,sampling,exit_code,expected in (
                (True,True,0,'converged'),(True,False,0,'completed_gate_failed'),
                (False,False,3,'failed')):
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as folder:
                repo=Path(folder); scripts=repo/'scripts'; scripts.mkdir()
                script=scripts/'run_adaptive_full_case.py'
                shutil.copyfile(Path(__file__).with_name(script.name),script)
                package=repo/'code/rydberg_kagome'; package.mkdir(parents=True)
                (package/'__init__.py').write_text('')
                (package/'adaptive_memory.py').write_text('GIB=2**30\ndef snapshot(): return dict(available_bytes=128*GIB)\n')
                (package/'results.py').write_text('def source_hashes(): return {}\n')
                (package/'__main__.py').write_text(f'''import sys,json
from pathlib import Path
def main():
    root=Path(sys.argv[sys.argv.index('--output')+1]); root.mkdir(exist_ok=True)
    (root/'resume.json').write_text('{{}}')
    if {exit_code}: raise SystemExit({exit_code})
    (root/'result.json').write_text(json.dumps(dict(solver_pass={solver},sampling_moment_check={sampling},bond_dimension_verified=False,rg_converged=False,production_qualified=False)))
''')
                out=repo/'output'
                subprocess.run([sys.executable,'-B',str(script),'--root',str(out),
                    '--model-cache',str(repo/'cache')],check=True,capture_output=True,timeout=15)
                report=json.loads((out/'status.json').read_text())
                self.assertTrue(report['finished']); self.assertEqual(report['state'],expected)
                self.assertEqual(report['full_case_gate_passed'],expected=='converged')
                self.assertEqual(report['exit_code'],exit_code)
                self.assertGreater(json.loads((out/'worker-resources.json').read_text())['peak_rss_kib'],0)
                subprocess.run([sys.executable,'-B',str(script),'--root',str(out),
                    '--model-cache',str(repo/'cache'),'--resume','--threads','2'],
                    check=True,capture_output=True,timeout=15)
                resumed=json.loads((out/'status.json').read_text())
                self.assertEqual(resumed['threads'],2)
                self.assertEqual(resumed['original_started'],report['started'])
                self.assertEqual(resumed['resumed_from_worker'],report['worker'])
                self.assertEqual(resumed['state'],expected)
                self.assertEqual(len(list(out.glob('*-worker.log'))),1)


if __name__=='__main__': unittest.main()
